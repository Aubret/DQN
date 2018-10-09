package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.Mlp;
import fr.univlyon1.networks.StateApproximator;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;

@Getter
@Setter
public class TDLstm<A> extends TD<A> {

    /*
    Attention classe pas maintenue !!!! Utiliser le TDLSTM 2D. Seules les parties communes sont ok
     */
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


        /*this.targetObservationApproximator = this.observationApproximator.clone(false);
        this.observationApproximator = observationApproximator ;
        this.cloneObservationApproximator = cloneObservationApproximator ;*/
        this.targetObservationApproximator = observationApproximator ;
        this.observationApproximator = this.targetObservationApproximator.clone(true);
        this.cloneObservationApproximator = cloneObservationApproximator ;

    }

    @Override
    public INDArray behave(INDArray input){
        if(this.lastInteraction != null) {
            INDArray act = (INDArray)this.learning.getActionSpace().mapActionToNumber(this.lastInteraction.getAction());
            INDArray actualState = this.observationApproximator.getOneResult(Nd4j.concat(1,this.lastInteraction.getObservation(),act));
            INDArray state_observation = Nd4j.concat(1,actualState,input);
            return (INDArray)this.learning.getPolicy().getAction(state_observation,this.informations);
        }else{
            System.out.println("behave pas bon ! faut au moins une action random");
            return null ;
        }

    }

    @Override
    public void evaluate(INDArray input, Double reward) { // Store transistions
        if(this.lastInteraction != null) { // Avoir des interactions complètes
            /*if(this.lastInteraction.getObservation().getDouble(4)==-0.5){
                INDArray act = (INDArray)(this.learning.getActionSpace().mapActionToNumber(this.lastInteraction.getAction()));
                if(act.getDouble(1) < -0.5){
                    reward += 0.3 ;
                }
            }*/
            this.lastInteraction.setSecondObservation(input);
            this.lastInteraction.setReward(reward);
            this.experienceReplay.addInteraction(this.lastInteraction);
        }
    }

    /**
     * Extract mini batch
     */
    @Override
    public void learn(){
        Object state2 = this.cloneObservationApproximator.getMemory();
        //this.state = this.observationApproximator.getMemory();
        this.state = this.targetObservationApproximator.getMemory();
        if(this.replay)
            this.learn_replay();
        this.informations.setModified(true);
        //this.observationApproximator.setMemory(this.state);
        this.cloneObservationApproximator.setMemory(state2);
    }

    // Désuete et non mis à jour
    protected void learn_replay(){
        int numRows = Math.min(this.experienceReplay.getSize(),this.batchSize);
        int size = this.learning.getObservationSpace().getShape()[0];
        if(numRows < 1 ) {
            return;
        }
        for(int i = 0;i < this.iterations ; i++){
            ArrayList<INDArray> inputsArray = new ArrayList<>();
            INDArray secondObservations = Nd4j.zeros(numRows,size);
            INDArray rewards = Nd4j.zeros(numRows,1);
            //INDArray inputs = Nd4j.zeros(dimensions);

            INDArray actions = Nd4j.zeros(numRows,this.learning.getActionSpace().getSize());
            ArrayList<Integer> forwardsNumbers = new ArrayList<>();
            int forward = 0 ;
            for (int r = 0; r < numRows; r++) {
                if (this.experienceReplay.initChoose()) {
                    // choix des interactions
                    Interaction<A> lastInteraction = null;
                    Interaction<A> interaction = this.experienceReplay.chooseInteraction();
                    ArrayList<Interaction<A>> observations = new ArrayList<>();
                    while (interaction != null) {
                        observations.add(interaction);
                        lastInteraction = interaction;
                        interaction = this.experienceReplay.chooseInteraction();
                    }
                    //Préparation input
                    forwardsNumbers.add(this.experienceReplay.getForwardNumber());
                    forward = Math.max(forward,this.experienceReplay.getForwardNumber());
                    int[] dimensions = new int[]{lastInteraction.getObservation().shape()[1], observations.size()};
                    inputsArray.add(Nd4j.zeros(dimensions));
                    for (int j = 0; j < observations.size(); j++) { // Inserstion des observations temporelles
                        INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(j)};
                        inputsArray.get(r).put(indexs, observations.get(j).getObservation());
                    }
                    //Préparation actions
                    INDArray action = (INDArray) this.learning.getActionSpace().mapActionToNumber(lastInteraction.getAction());
                    INDArrayIndex[] index =new INDArrayIndex[]{NDArrayIndex.point(r),NDArrayIndex.all()};
                    actions.put(index,action);
                    //Labellisation des secondes observations
                    secondObservations.put(index,lastInteraction.getSecondObservation());
                    rewards.put(index,lastInteraction.getSecondObservation());

                }else{
                    return ;
                }
            }
            INDArray inputs = Nd4j.zeros(numRows,size,forward);// On avait besoin de la taille maximale du forward
            INDArray masks = Nd4j.zeros(numRows,forward);
            INDArray maskLabel = Nd4j.zeros(numRows,forward);
            for(int in = 0; in < inputsArray.size() ; in++){ // Inserstion de chaque batch de données temporelles
                INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.point(in),NDArrayIndex.all(),NDArrayIndex.interval(0,inputsArray.get(in).size(1))};
                inputs.put(indexs, inputsArray.get(in));
                INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(in),NDArrayIndex.interval(0,inputsArray.get(in).size(1))};
                masks.put(indexMask, Nd4j.ones(inputsArray.get(in).size(1)));
                INDArrayIndex[] indexMaskLabel = new INDArrayIndex[]{NDArrayIndex.point(in),NDArrayIndex.point(inputsArray.get(in).size(1)-1)};
                maskLabel.put(indexMaskLabel, Nd4j.ones(1));
            }
            INDArray labels = this.labelize(secondObservations,rewards);
            // Apprentissage : besoin de l'état
            INDArray state = this.observationApproximator.forwardLearn(inputs, null, numRows,masks,maskLabel);
            INDArray state_label = Nd4j.concat(1,state,inputs);
            int sizeObservation = state_label.size(1);


            //state_label= state_label.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(state_label.size(2)-1));

            INDArray inputCritics = Nd4j.concat(1, state_label, actions);
            INDArray epsilon = this.learn_critic(inputCritics, labels, numRows,sizeObservation);

            //Apprentissage politique
            int sizeAction = this.learning.getActionSpace().getSize();
            /*INDArray epsilonActor = */this.learn_actor(state_label, sizeObservation, sizeAction, 1); // Important entre la propagation de l'observation et la backpropagation du gradient

            //Apprentissage des observations
            //INDArray epsilonObservationAct = epsilonActor.get(NDArrayIndex.all(), NDArrayIndex.interval(0, sizeObservation));
            INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, sizeObservation));
            INDArray labelObservation = Nd4j.zeros(numRows,this.observationApproximator.numOutput(),forward);
            for(int k = 0 ; k < forwardsNumbers.size(); k++){
                INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(forwardsNumbers.get(k)-1)};
                labelObservation.put(indexs,epsilonObservation);
            }
            this.learn_observator(inputs, labelObservation, numRows, actions, state_label, labels);
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
