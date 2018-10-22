package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.LSTMMeanPooling;
import fr.univlyon1.networks.StateApproximator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;

public class TDLstmFilter<A> extends TDLstm<A> {
    public TDLstmFilter(double gamma, Learning<A> learning, SequentialExperienceReplay<A> experienceReplay, int iterations, int batchSize, Approximator criticApproximator, Approximator cloneCriticApproximator, StateApproximator observationApproximator, StateApproximator cloneObservationApproximator) {
        super(gamma, learning, experienceReplay, iterations, batchSize, criticApproximator, cloneCriticApproximator, observationApproximator, cloneObservationApproximator);
    }


    /* Pas plus de 1 backward attention à casue de secondObservation2

     */
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


            INDArray rewards = Nd4j.zeros(backward,1);
            INDArray gammas = Nd4j.zeros(backward,1);

            INDArray actions = Nd4j.zeros(backward,this.learning.getActionSpace().getSize());
            INDArray inputs2 = Nd4j.zeros(backward,this.learning.getObservationSpace().getShape()[0]);// On avait besoin de la taille maximale du forward


            INDArray inputs = Nd4j.zeros(totalBatchs,size,forwardInputs);// On avait besoin de la taille maximale du forward
            //INDArray secondObservations3 = Nd4j.zeros(totalBatchs,size,forward);// On avait besoin de la taille maximale du forward
            INDArray inputsTarget = Nd4j.zeros(totalBatchs,size,forwardInputs);// On avait besoin de la taille maximale du forward


            INDArray masks = Nd4j.zeros(totalBatchs,forwardInputs);
            INDArray masksTarget = Nd4j.zeros(totalBatchs,forwardInputs);

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
                int cursorTarget = forwardInputs-1 ;
                for(int temporal =forward -1 ; temporal >= start; temporal--){ // INsertion temporelles
                    int cursorForward = forward-1-temporal ;
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
                        if(interact.getIdObserver() != observations.get(observations.size()-1).getIdSecondObserver()) {
                            inputsTarget.put(new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.point(cursorTarget)}, Nd4j.concat(1, interact.getObservation(), action));
                            INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(batch),NDArrayIndex.point(cursorTarget)};
                            masksTarget.put(indexMask, Nd4j.ones(1));
                            cursorTarget-- ;
                        }

                        //seconds observations for labelisation make observations 3
                        //secondObservations3.put(indexs,Nd4j.concat(1,interact.getSecondObservation(),action2));

                        //mask index
                        INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(batch),NDArrayIndex.point(temporal)};
                        masks.put(indexMask, Nd4j.ones(1));
                    }
                }
            }
            /*System.out.println("-----");
            System.out.println(masks);
            System.out.println(maskLabel);*/
            this.experienceReplay.setConstructedData(inputs); // save for self supervised learning
            // Apprentissage : besoin de l'état
            INDArray targetState = this.targetObservationApproximator.forwardLearn(inputsTarget,null,inputs.size(0),masksTarget,maskLabel);
            INDArray state = this.observationApproximator.forwardLearn(inputs, null, totalBatchs,masks,maskLabel);
            INDArray state_label = Nd4j.concat(1,state,inputs2);
            //INDArray targetState_label = Nd4j.concat(1,targetState,inputs2);

            //System.out.println("--------");

            int sizeObservation = state_label.size(1);
            //Commencement de l'apprentissage, labellisation
            //this.targetObservationApproximator.setMaskLabel(maskLabel);
            //INDArray obs1 = inputs.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(0));
            //INDArray labels = this.multistepLabelize(secondObservations3,rewards,secondObservations2,gammas, forwardInputs,totalBatchs,masks,maskLabel); // A faire après le forard learn pour avoir la bonne mémoire
            //INDArray labels = this.labelize(secondObservations,rewards,secondObservations2,gammas); // A faire après le forard learn pour avoir la bonne mémoire
            INDArray labels = this.labelizeFullTarget(inputsTarget, secondObservations,rewards,secondObservations2,gammas,masks,maskLabel );

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
}
