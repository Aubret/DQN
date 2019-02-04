package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.configurations.SavesLearning;
import fr.univlyon1.environment.states.AllHiddenState;
import fr.univlyon1.environment.states.HiddenState;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.*;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.workspace.WorkspaceMgr;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Getter
@Setter
@Slf4j
/**
 * LSTM with actor critic, but we send a whole trajectory into the lstm directly.
 * We have to construct the paddings and assign labels to trajectories.
 */
public class TDLstm2D<A> extends TDLstm<A> {
    protected SavesLearning savelearning ;

    public TDLstm2D(double gamma, Learning<A> learning, SequentialExperienceReplay<A> experienceReplay, int iterations, int batchSize,Approximator criticApproximator, Approximator cloneCriticApproximator, StateApproximator observationApproximator,StateApproximator cloneObservationApproximator) {
        super(gamma, learning,experienceReplay,iterations,batchSize,criticApproximator,cloneCriticApproximator,observationApproximator,cloneObservationApproximator);
        this.savelearning = new SavesLearning();

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
            while(backward < numRows) {
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
            //INDArray inputsTarget = Nd4j.zeros(totalBatchs,size,forwardInputs+1);// On avait besoin de la taille maximale du forward


            INDArray masks = Nd4j.zeros(totalBatchs,forwardInputs);
            INDArray maskLabel = Nd4j.zeros(totalBatchs*forwardInputs,1);
            int cursorBackward = 0 ;
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
                        cursorBackward++ ;
                    }

                    //Seule la dernière donnée des inputs est labellisée
                    //if(temporal == forwardInputs-1){
                    if(temporal >= forwardInputs-numBackwards+1 && temporal < forwardInputs){
                        maskLabel.put(new INDArrayIndex[]{NDArrayIndex.point(totalBatchs*temporal + batch),NDArrayIndex.all()},Nd4j.ones(1));
                        //Labellisation des secondes observations
                        secondObservations.put(new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all()/*, NDArrayIndex.point(temporal)*/},Nd4j.concat(1,interact.getSecondObservation(),action2));
                    }

                    if(temporal < forwardInputs) {
                        INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.point(temporal)};
                        inputs.put(indexs, Nd4j.concat(1, interact.getObservation(), action));
                        secondObservations3.put(indexs,Nd4j.concat(1, interact.getSecondObservation(), action2));

                        //mask index
                        INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(batch),NDArrayIndex.point(temporal)};
                        masks.put(indexMask, Nd4j.ones(1));
                    }
                    cursorForward++ ;
                }
            }
            this.experienceReplay.setConstructedData(inputs); // save for self supervised learning
            // Apprentissage : besoin de l'état
            INDArray targetState = this.targetObservationApproximator.forwardLearn(inputs,null,(int)inputs.size(0),masks,maskLabel);
            INDArray state = this.observationApproximator.forwardLearn(inputs, null, totalBatchs,masks,maskLabel);
            INDArray state_label = Nd4j.concat(1,state,inputs2);
            INDArray targetState_label = Nd4j.concat(1,targetState,inputs2);


            long sizeObservation = state_label.size(1);
            //Commencement de l'apprentissage, labellisation
            //this.targetObservationApproximator.setMaskLabel(maskLabel);
            //INDArray obs1 = inputs.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(0));
            //INDArray labels = this.multistepLabelize(secondObservations3,rewards,secondObservations2,gammas, forwardInputs,totalBatchs,masks,maskLabel); // A faire après le forard learn pour avoir la bonne mémoire
            //INDArray labels = this.labelize(secondObservations,targetState_label, rewards,secondObservations2,gammas); // A faire après le forard learn pour avoir la bonne mémoire
            INDArray labels = this.labelizeFullTarget(inputs, secondObservations,rewards,secondObservations2,gammas,masks,maskLabel );

            //Apprentissage critic
            INDArray inputCritics = Nd4j.concat(1, state_label,actions);
            INDArray epsilon = this.learn_critic(inputCritics, labels, totalBatchs,sizeObservation);
            this.savelearning.add(this.criticApproximator.getScore());
            INDArray score = this.criticApproximator.getScoreArray();
            this.experienceReplay.setError(score, backwardsNumber, backward, total);

            //erreur sur la politique
            //INDArray epsilonAction = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(sizeObservation, sizeObservation+this.learning.getActionSpace().getSize()));
            //INDArray epsilonActor = this.learning.getApproximator().error(state_label, epsilonAction,totalBatchs*forward);

            //Apprentissage politique
            int sizeAction = this.learning.getActionSpace().getSize();
            this.learn_actor(targetState_label, state_label, sizeObservation, sizeAction, totalBatchs); // Important entre la propagation de l'observation et la backpropagation du gradient

            //INDArray epsilonObservationCrit = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.observationApproximator.numOutput()));
            INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.observationApproximator.numOutput()));
            //INDArray epsilonObservationAct = epsilonActor.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.observationApproximator.numOutput()));
            //INDArray epsilonObservation = epsilonObservationCrit.addi(epsilonObservationAct);
            this.learn_observator(inputs, epsilonObservation, totalBatchs*forwardInputs, actions, inputs2, labels);
            this.cpt_time++;
        }
    }


    protected INDArray learn_actor(INDArray observations, INDArray observationsCritic, long sizeObservation, int numColumns, int numRows){
        INDArray action = this.learning.getApproximator().getOneResult(observations); // L'action du policy networks
        this.informations.setEvaluatedInputs(observations);
        this.informations.setEvaluatedActions(action);
        INDArray inputAction = Nd4j.concat(1, observationsCritic, action);
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
            log.info(mean + " -- " + cpt);
        }
        return eps;
    }

    protected INDArray labelize2(INDArray secondObservations ,INDArray rewards,INDArray secondObservations2 ,INDArray gammas){
        // Les états précédents sont dans la mémoire de l'approximateur
        this.targetObservationApproximator.clear();

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

    protected INDArray multistepLabelize(INDArray secondObservations ,INDArray rewards,INDArray secondObservations2,INDArray gammas, int forward,int batchsize,INDArray mask,INDArray masklabel) {
        // Les états précédents sont dans la mémoire de l'approximateur
        this.targetObservationApproximator.clear();

        AllHiddenState ahs = (AllHiddenState) this.observationApproximator.allMemory() ;
        ArrayList<INDArray[]> states = ahs.getTimeStates() ;
        ArrayList<INDArray[]> acts = ahs.getPrevActs() ;
        INDArray targets = Nd4j.zeros(batchsize * forward, this.observationApproximator.numOutput());
        for(int k = 0 ; k < forward ; k++) { // chaque timestep
            ArrayList<Map<String, INDArray>> hidden = new ArrayList<>();

            for (int i = 0; i < states.size(); i++) { // chaque layer
                Map<String, INDArray> statemap = new ConcurrentHashMap();
                INDArray[] statesOflayers = states.get(i);
                if(statesOflayers.length >= forward - k){
                    int index = statesOflayers.length - forward + k ;
                    statemap.put("prevMem",states.get(i)[index]);
                    statemap.put("prevAct",acts.get(i)[index]);
                }else{
                    statemap.put("prevMem",Nd4j.zeros(batchsize,this.observationApproximator.numOutput()));
                    statemap.put("prevAct",Nd4j.zeros(batchsize,this.observationApproximator.numOutput()));
                }
                hidden.add(statemap);
            }
            HiddenState hiddenState = new HiddenState(hidden);
            this.targetObservationApproximator.setMemory(hiddenState);

            INDArrayIndex[] index = new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(k)};
            INDArray state = this.targetObservationApproximator.getOneResult(secondObservations.get(index));
            targets.put(new INDArrayIndex[]{NDArrayIndex.interval(k*batchsize,(k+1)*batchsize),NDArrayIndex.all()},state.dup());

        }
        INDArray aggregateState = ((LSTM2D)targetObservationApproximator).crop3dData(targets,masklabel);
        INDArray stateLabel = Nd4j.concat(1,aggregateState,secondObservations2);

        INDArray action = this.targetActorApproximator.getOneResult(stateLabel);
        INDArray entryCriticTarget = Nd4j.concat(1,stateLabel, action) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        res = res.muli(gammas);
        res.addi(rewards) ;
        return res;
        //return res ;
    }

    protected INDArray labelize(INDArray secondObservations , INDArray stateLabelT, INDArray rewards,INDArray secondObservations2,INDArray gammas){
        // Les états précédents sont dans la mémoire de l'approximateur
        this.targetObservationApproximator.clear(); // recheck le fonctionnement de la mémory
        this.targetObservationApproximator.setMemory(this.observationApproximator.getSecondMemory());

        INDArray state = this.targetObservationApproximator.getOneResult(secondObservations);
        INDArray stateLabel = Nd4j.concat(1,state,secondObservations2);

        INDArray action = this.targetActorApproximator.getOneResult(stateLabel);
        INDArray entryCriticTarget = Nd4j.concat(1,stateLabelT, action) ;
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
        return res ;
    }



}
