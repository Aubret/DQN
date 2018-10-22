package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.configurations.SavesLearning;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;
import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.memory.OneVehicleSequentialExperienceReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.LSTMMeanPooling;
import fr.univlyon1.networks.Mlp;
import fr.univlyon1.networks.StateApproximator;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Stack;

@Getter
@Setter
public class TDLstm<A> extends TD<A> {

    protected SavesLearning savelearning ;

    protected StateApproximator observationApproximator ;
    protected StateApproximator cloneObservationApproximator ;
    protected StateApproximator targetObservationApproximator ;
    protected Approximator targetActorApproximator ;
    protected Approximator criticApproximator ;
    protected Approximator targetCriticApproximator ;
    protected Approximator cloneCriticApproximator ;
    public boolean replay = true ;

    protected Object state ;

    protected double cpt = 0 ;
    protected double cumulScoreUp=0;
    protected double cumulObservation= 0;
    protected double cumulOnlyObservation= 0;

    protected int time =20;
    protected int cpt_time = 0 ;
    protected boolean t = true ;

    protected SequentialExperienceReplay<A> experienceReplay;
    protected int iterations ;
    protected int batchSize ;

    public TDLstm(double gamma, Learning<A> learning, SequentialExperienceReplay<A> experienceReplay, int iterations, int batchSize,Approximator criticApproximator, Approximator cloneCriticApproximator, StateApproximator observationApproximator,StateApproximator cloneObservationApproximator) {
        super(gamma, learning);
        this.iterations = iterations ;
        this.batchSize = batchSize ;
        this.experienceReplay = experienceReplay ;

        //this.targetObservationApproximator = observationApproximator.clone(false);
        this.targetActorApproximator = this.learning.getApproximator().clone(false);
        this.criticApproximator = criticApproximator ;
        this.approximator = criticApproximator;
        this.targetCriticApproximator = criticApproximator.clone(false);
        this.cloneCriticApproximator = cloneCriticApproximator ; // Le clône permet de traiter deux fontions de pertes différentes.


        this.targetObservationApproximator = observationApproximator.clone(false);
        this.observationApproximator = observationApproximator ;
        this.cloneObservationApproximator = cloneObservationApproximator ;
        /*this.targetObservationApproximator = observationApproximator ;
        this.observationApproximator = this.targetObservationApproximator.clone(true);
        this.cloneObservationApproximator = cloneObservationApproximator ;*/
        this.savelearning = new SavesLearning();


    }

    @Override
    public INDArray behave(INDArray input){
        if(this.lastInteraction != null) {
            INDArray act = (INDArray)this.learning.getActionSpace().mapActionToNumber(this.lastInteraction.getAction());
            //INDArray actualState = this.targetObservationApproximator.getOneResult(Nd4j.concat(1,this.lastInteraction.getObservation(),act));
            Stack<Replayable<A>> lastInteractions = this.experienceReplay.lastInteraction() ;
            Interaction<A> inter = (Interaction<A>)lastInteractions.pop();
            if(this.experienceReplay instanceof OneVehicleSequentialExperienceReplay)
                this.observationApproximator.clear();

            while(!lastInteractions.isEmpty()){
                this.observationApproximator.getOneResult(Nd4j.concat(1,inter.getObservation(),(INDArray)this.learning.getActionSpace().mapActionToNumber(inter.getAction())));
                inter = (Interaction<A>)lastInteractions.pop();
            }
            INDArray actualState = this.observationApproximator.getOneResult(Nd4j.concat(1,inter.getObservation(),(INDArray)this.learning.getActionSpace().mapActionToNumber(inter.getAction())));

            //INDArray actualState = this.observationApproximator.getOneResult(Nd4j.concat(1,this.lastInteraction.getObservation(),act));
            INDArray state_observation = Nd4j.concat(1,actualState,input);
            return (INDArray)this.learning.getPolicy().getAction(state_observation,this.informations);
        }else{
            System.out.println("behave pas bon ! faut au moins une action random");
            return null ;
        }

    }

    @Override
    public void evaluate(Observation input, Double reward, Double time) { // Store transistions
        super.evaluate(input,reward,time);
        if(this.lastInteraction != null) { // Avoir des interactions complètes
            this.experienceReplay.addInteraction(this.lastInteraction);
        }
    }

    /**
     * Extract mini batch
     */
    @Override
    public void learn(){
        //Object state2 = this.cloneObservationApproximator.getMemory();
        this.state = this.observationApproximator.getMemory();
        this.experienceReplay.setMinForward(this.learning.getConf().getForward()+1);
        if(this.replay)
            this.learn_replay();
        this.informations.setModified(true);
        this.experienceReplay.setMinForward(this.learning.getConf().getForward());
        this.observationApproximator.setMemory(this.state);
        //this.cloneObservationApproximator.setMemory(state2);
    }

    protected void learn_replay(){
        int numRows = Math.min(this.experienceReplay.getSize(),this.batchSize);
        int size = this.learning.getObservationSpace().getShape()[0]+this.learning.getActionSpace().getSize();
        if(numRows < 1 ) {
            return;
        }
        //Récupération de toutes les interactions
        for(int i = 0;i < this.iterations ; i++){
            if(i > 0){
                this.epoch() ;
            }
            ArrayList<ArrayList<Interaction<A>>> total = new ArrayList<>();
            ArrayList<Integer> backwardsNumber = new ArrayList<>(); // nombre de backward pour chaque batch
            int forward = 0 ; // Maximum de taille de séquence temporelle
            int backward = 0 ; // Total de données labellisées, donc total de backpropagation
            while(backward <= numRows) {
                if (this.experienceReplay.initChoose()) {
                    // choix des interactions
                    Interaction<A> interaction = this.experienceReplay.chooseInteraction();
                    ArrayList<Interaction<A>> observations = new ArrayList<>();
                    while (interaction != null) {
                        observations.add(interaction);
                        interaction = this.experienceReplay.chooseInteraction();
                    }
                    forward = Math.max(forward,this.experienceReplay.getForwardNumber());
                    total.add(observations);
                    int back= this.experienceReplay.getBackpropNumber();
                    backwardsNumber.add(back);
                    backward+=back;
                }else{
                    if(total.size() == 0){
                        return ;
                    }else{
                        break ;
                    }
                }
            }
            int forwardInputs = forward-1 ;
            //contruction des INDArrays
            int totalBatchs = total.size();
            backward = backward - totalBatchs;
            //INDArray secondObservations = Nd4j.zeros(totalBatchs,size,forward);
            INDArray secondObservations = Nd4j.zeros(totalBatchs,size);
            INDArray secondObservations2 = Nd4j.zeros(backward,this.learning.getObservationSpace().getShape()[0]);// On avait besoin de la taille maximale du forward
            INDArray secondObservations3 = Nd4j.zeros(totalBatchs,size,forwardInputs);


            INDArray rewards = Nd4j.zeros(backward,1);
            INDArray gammas = Nd4j.zeros(backward,1);

            INDArray actions = Nd4j.zeros(backward,this.learning.getActionSpace().getSize());
            INDArray inputs2 = Nd4j.zeros(backward,this.learning.getObservationSpace().getShape()[0]);// On avait besoin de la taille maximale du forward


            INDArray inputs = Nd4j.zeros(totalBatchs,size,forwardInputs);// On avait besoin de la taille maximale du forward
            //INDArray secondObservations3 = Nd4j.zeros(totalBatchs,size,forward);// On avait besoin de la taille maximale du forward
            INDArray inputsTarget = Nd4j.zeros(totalBatchs,size,forwardInputs+1);// On avait besoin de la taille maximale du forward


            INDArray masks = Nd4j.zeros(totalBatchs,forwardInputs);
            INDArray maskLabel ;
            if(observationApproximator instanceof LSTMMeanPooling){
                maskLabel = Nd4j.ones(totalBatchs,1);
            }else
                maskLabel = Nd4j.zeros(totalBatchs*forwardInputs,1);
            int cursorBackward = 0 ;
            //System.out.println(forward);
            for(int batch = 0 ; batch < total.size() ; batch++){ // Insertion des batchs
                ArrayList<Interaction<A>> observations = total.get(batch);
                int numberObservation = observations.size() ;//Nombre données temporelles
                int numBackwards = backwardsNumber.get(batch); // Nombre de backwards parmi ces données
                int start = forward - numberObservation ;
                int cursorForward = 0 ;
                for(int temporal =0 ; temporal < forward; temporal++){ // INsertion temporelles
                    if(temporal < start)
                        continue ;
                    Interaction<A> interact = observations.get(cursorForward);
                    //labels globaux
                    int indice = - numberObservation +numBackwards + cursorForward;
                    //int indice = - numberObservation +numBackwards + temporal;
                    INDArray action = (INDArray) this.learning.getActionSpace().mapActionToNumber(interact.getAction());
                    INDArray action2 = (INDArray) this.learning.getActionSpace().mapActionToNumber(interact.getSecondAction());
                    if(indice > 0){ // de >= à >0
                        //Préparation actions
                        actions.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward), NDArrayIndex.all()}, action);
                        //inputs observations pour la concaténation
                        inputs2.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward), NDArrayIndex.all()}, interact.getObservation());
                        //secondes observations pour la conténation de la labellisation
                        secondObservations2.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward), NDArrayIndex.all()},interact.getSecondObservation());
                        //rewards
                        rewards.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward),NDArrayIndex.all()}, interact.computeReward());
                        //gammas
                        gammas.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward),NDArrayIndex.all()}, interact.computeGamma());
                        //System.out.println(interact.computeGamma());
                        cursorBackward++ ;
                    }

                    //Seule la dernière donnée des inputs est labellisée
                    //if(temporal == forwardInputs-1){
                    if(temporal >= forwardInputs-numBackwards+1 && temporal < forwardInputs){
                        //if(temporal == numberObservation-2){
                        //Label du mask
                        //maskLabel.put(new INDArrayIndex[]{NDArrayIndex.point(batch*forward + temporal),NDArrayIndex.all()},Nd4j.ones(1));
                        if(!(observationApproximator instanceof LSTMMeanPooling))
                            maskLabel.put(new INDArrayIndex[]{NDArrayIndex.point(totalBatchs*temporal + batch),NDArrayIndex.all()},Nd4j.ones(1));
                        //Labellisation des secondes observations
                        secondObservations.put(new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all()/*, NDArrayIndex.point(temporal)*/},Nd4j.concat(1,interact.getSecondObservation(),action2));

                        //secondObservations.
                    }

                    if(temporal < forwardInputs) {
                        //if(temporal != numberObservation-1) {
                        //inputs observations
                        INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.point(temporal)};
                        inputs.put(indexs, Nd4j.concat(1, interact.getObservation(), action));
                        //secondObservations3.put(indexs,Nd4j.concat(1, interact.getSecondObservation(), action2));

                        //seconds observations for labelisation make observations 3
                        //secondObservations3.put(indexs,Nd4j.concat(1,interact.getSecondObservation(),action2));

                        //mask index
                        INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(batch),NDArrayIndex.point(temporal)};
                        masks.put(indexMask, Nd4j.ones(1));
                    }
                    cursorForward++ ;
                }
            }
            /*System.out.println("-----");
            System.out.println(masks);
            System.out.println(maskLabel);*/
            this.experienceReplay.setConstructedData(inputs); // save for self supervised learning
            // Apprentissage : besoin de l'état
            //INDArray targetState = this.targetObservationApproximator.forwardLearn(inputs,null,inputs.size(0),masks,maskLabel);
            INDArray state = this.observationApproximator.forwardLearn(inputs, null, totalBatchs,masks,maskLabel);
            INDArray state_label = Nd4j.concat(1,state,inputs2);
            //INDArray targetState_label = Nd4j.concat(1,targetState,inputs2);

            //System.out.println("--------");

            int sizeObservation = state_label.size(1);
            //Commencement de l'apprentissage, labellisation
            //this.targetObservationApproximator.setMaskLabel(maskLabel);
            //INDArray obs1 = inputs.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(0));
            //INDArray labels = this.multistepLabelize(secondObservations3,rewards,secondObservations2,gammas, forwardInputs,totalBatchs,masks,maskLabel); // A faire après le forard learn pour avoir la bonne mémoire
            INDArray labels = this.labelize(secondObservations,rewards,secondObservations2,gammas); // A faire après le forard learn pour avoir la bonne mémoire
            //INDArray labels = this.labelizeFullTarget(inputs, secondObservations,rewards,secondObservations2,gammas,masks,maskLabel );

            //Apprentissage critic
            INDArray inputCritics = Nd4j.concat(1, state_label,actions);
            //System.out.println(inputCritics);
            INDArray epsilon = this.learn_critic(inputCritics, labels, totalBatchs,sizeObservation);
            this.savelearning.add(this.criticApproximator.getScore());
            INDArray score = this.criticApproximator.getScoreArray();
            this.experienceReplay.setError(score, backwardsNumber, backward, total);

            //erreur sur la politique
            //INDArray epsilonAction = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(sizeObservation, sizeObservation+this.learning.getActionSpace().getSize()));
            //INDArray epsilonActor = this.learning.getApproximator().error(state_label, epsilonAction,totalBatchs*forward);

            //Apprentissage politique
            int sizeAction = this.learning.getActionSpace().getSize();
            this.learn_actor(state_label, sizeObservation, sizeAction, totalBatchs); // Important entre la propagation de l'observation et la backpropagation du gradient

            //INDArray epsilonObservationCrit = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.observationApproximator.numOutput()));
            INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.observationApproximator.numOutput()));
            //INDArray epsilonObservationAct = epsilonActor.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.observationApproximator.numOutput()));
            //INDArray epsilonObservation = epsilonObservationCrit.addi(epsilonObservationAct);
            this.learn_observator(inputs, epsilonObservation, totalBatchs*forwardInputs, actions, inputs2, labels);
            this.cpt_time++;
        }
    }

    protected void learn_observator(INDArray inputs, INDArray epsilonObservation, int numRows,INDArray action,INDArray inputs2,INDArray labels){
        this.observationApproximator.learn(inputs,epsilonObservation,numRows);
        if(this.cpt_time%this.time == 0){
            INDArray firstval = ((Mlp) this.criticApproximator).getValues().detach();
            INDArray s1 = firstval.sub(labels);
            Double val1 = s1.muli(s1).meanNumber().doubleValue();

            INDArray res = this.observationApproximator.getOneTrainingResult(inputs);
            INDArray inputAction = Nd4j.concat(1,res,inputs2,action);
            INDArray newVal = this.criticApproximator.getOneResult(inputAction);
            INDArray s2 = newVal.sub(labels);
            Double val2 =s2.muli(s2).meanNumber().doubleValue();
            double meanScore = val1-val2 ;
            cumulObservation+=meanScore ;
            //cumulOnlyObservation+=(val3-val2);
            System.out.println(meanScore + " -- " +(cumulObservation-cumulScoreUp));
            //System.out.println((val3-val2) + " -- " + cumulOnlyObservation);

        }

    }

    protected INDArray learn_critic(INDArray inputs, INDArray labels, int numRows,int sizeObservation){


        //INDArray epsilonObsAct = (INDArray)this.criticApproximator.learn(inputs, labels, numRows); // Critic learning
        INDArray epsilon = (INDArray)this.criticApproximator.learn(inputs, labels, numRows);// Critic learning
        INDArray scores = this.criticApproximator.getScoreArray();
        this.experienceReplay.setError(scores);

        if(this.cpt_time%this.time == 0){
            System.out.println("-------------");
            INDArray firstval = ((Mlp) this.criticApproximator).getValues().detach();
            INDArray s1 = firstval.sub(labels);
            Double val1 = s1.muli(s1).meanNumber().doubleValue();
            INDArray newVal = this.criticApproximator.getOneResult(inputs);
            INDArray s2 = newVal.sub(labels);
            Double val2 =s2.muli(s2).meanNumber().doubleValue();
            double meanScore = val1-val2 ;
            cumulScoreUp+=meanScore ;
            System.out.println(meanScore + " -- " +cumulScoreUp);

        }
        //INDArray epsAct = INDArrayIndex.interval(sizeObservation, epsilonAction.size(1));

        return epsilon;
    }

    protected INDArray learn_actor(INDArray observations, int sizeObservation, int numColumns, int numRows){
        INDArray action = this.learning.getApproximator().getOneResult(observations); // L'action du policy networks
        this.informations.setEvaluatedInputs(observations);
        this.informations.setEvaluatedActions(action);
        INDArray inputAction = Nd4j.concat(1, observations, action);
        this.cloneCriticApproximator.setParams(this.criticApproximator.getParams()); // Dupliquer les paramètres
        INDArray epsilonObsAct = this.cloneCriticApproximator.error(inputAction, Nd4j.create(new double[]{0}), numRows); // erreur
        INDArray epsilonAction = epsilonObsAct.get(NDArrayIndex.all(), NDArrayIndex.interval(sizeObservation, sizeObservation+numColumns));
        INDArray old =null;
        if(this.cpt_time%this.time == 0) {
            old = this.cloneCriticApproximator.getOneResult(inputAction);
        }
        INDArray eps = (INDArray) this.learning.getApproximator().learn(observations, epsilonAction, numRows); //Policy learning
        if(this.cpt_time%this.time == 0 ) {
            action = this.learning.getApproximator().getOneResult(observations); // L'action du policy network
            inputAction = Nd4j.concat(1, observations, action);
            INDArray intermediaire = this.cloneCriticApproximator.getOneResult(inputAction).subi(old);//must be positive
            Number mean = intermediaire.meanNumber();
            cpt += mean.doubleValue();
            System.out.println(mean + " -- " + cpt);
        }
        return eps;
    }


    protected INDArray labelize(INDArray secondObservations ,INDArray rewards){
        // Les états précédents sont dans la mémoire de l'approximateur

        this.targetObservationApproximator.setMemory(this.observationApproximator.getSecondMemory());
        INDArray stateLabel = this.targetObservationApproximator.getOneResult(secondObservations);
        INDArray action = this.targetActorApproximator.getOneResult(stateLabel);
        INDArray entryCriticTarget = Nd4j.concat(1,stateLabel, action) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        res = res.muli(this.gamma);
        res.addi(rewards) ;
        return res ;
    }

    protected INDArray labelize(INDArray secondObservations ,INDArray rewards,INDArray secondObservations2,INDArray gammas){
        // Les états précédents sont dans la mémoire de l'approximateur
        this.targetObservationApproximator.clear(); // recheck le fonctionnement de la mémory

        //System.out.println(((HiddenState)this.observationApproximator.getSecondMemory()).size());
        this.targetObservationApproximator.setMemory(this.observationApproximator.getSecondMemory());

        INDArray state = this.targetObservationApproximator.getOneResult(secondObservations);
        INDArray stateLabel = Nd4j.concat(1,state,secondObservations2);

        INDArray action = this.targetActorApproximator.getOneResult(stateLabel);
        INDArray entryCriticTarget = Nd4j.concat(1,stateLabel, action) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        res = res.muli(gammas);
        res.addi(rewards) ;
        return res ;
    }

    protected INDArray labelizeFullTarget(INDArray inputs, INDArray secondObservations ,INDArray rewards,INDArray secondObservations2,INDArray gammas,INDArray mask,INDArray masklabel){
        // Les états précédents sont dans la mémoire de l'approximateur
        //this.targetObservationApproximator.clear();
        this.targetObservationApproximator.setMemory(this.targetObservationApproximator.getSecondMemory());
        INDArray state = this.targetObservationApproximator.getOneResult(secondObservations);
        //INDArray state = this.targetObservationApproximator.forwardLearn(secondObservations,null, secondObservations.size(0), Nd4j.ones(secondObservations.size(0),1),Nd4j.ones(secondObservations.size(0),1));
        //INDArray state = this.targetObservationApproximator.getOneResult(secondObservations);
        INDArray stateLabel = Nd4j.concat(1,state,secondObservations2);

        INDArray action = this.targetActorApproximator.getOneResult(stateLabel);
        INDArray entryCriticTarget = Nd4j.concat(1,stateLabel, action) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        res = res.muli(gammas);
        res.addi(rewards) ;
        /*if(res.getDouble(0)==0.){
            System.out.println("gammes : "+gammas.getDouble(0));
        }*/
        return res ;
    }







    public void epoch() {
        double alphaActor = 0.01;
        double alphaCritic = 0.01;
        double alphaObserv = 0.01;
        this.targetActorApproximator.getParams().muli(1. - alphaActor).addi(this.learning.getApproximator().getParams().mul(alphaActor));
        this.targetCriticApproximator.getParams().muli(1. - alphaCritic).addi(this.criticApproximator.getParams().mul(alphaCritic));
        this.targetObservationApproximator.getParams().muli(1. - alphaObserv).addi(this.observationApproximator.getParams().mul(alphaObserv));
        //this.targetObservationApproximator.setParams(this.observationApproximator.getParams());
        //this.targetObservationApproximator.getParams().muli(1. - alpha).addi(this.observationApproximator.getParams().mul(alpha));

        /*targetActorApproximator.setParams(this.learning.getApproximator().getParams());
        targetCriticApproximator.setParams(this.criticApproximator.getParams());*/

        //targetObservationApproximator.setParams(this.observationApproximator.getParams());
    }
}
