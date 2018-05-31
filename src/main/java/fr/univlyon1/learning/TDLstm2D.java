package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.HiddenState;
import fr.univlyon1.environment.Interaction;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.LSTM;
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
public class TDLstm2D<A> extends TDLstm<A> {

    public TDLstm2D(double gamma, Learning<A> learning, SequentialExperienceReplay<A> experienceReplay, int iterations, int batchSize,Approximator criticApproximator, Approximator cloneCriticApproximator, StateApproximator observationApproximator,StateApproximator cloneObservationApproximator) {
        super(gamma, learning,experienceReplay,iterations,batchSize,criticApproximator,cloneCriticApproximator,observationApproximator,cloneObservationApproximator);
    }

    protected void learn_replay(){
        int numRows = Math.min(this.experienceReplay.getSize(),this.batchSize);
        int size = this.learning.getObservationSpace().getShape()[0];
        if(numRows < 1 ) {
            return;
        }
        //Récupération de toutes les interactions
        for(int i = 0;i < this.iterations ; i++){
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
                    return ;
                }
            }
            //contruction des INDArrays
            int totalBatchs = total.size();
            INDArray secondObservations = Nd4j.zeros(totalBatchs,size,forward);
            INDArray rewards = Nd4j.zeros(backward,1);
            INDArray actions = Nd4j.zeros(backward,this.learning.getActionSpace().getSize());
            INDArray inputs = Nd4j.zeros(totalBatchs,size,forward);// On avait besoin de la taille maximale du forward
            INDArray masks = Nd4j.zeros(totalBatchs,forward);
            INDArray maskLabel = Nd4j.zeros(totalBatchs*forward,1);
            int cursorBackward = 0 ;
            for(int batch = 0 ; batch < total.size() ; batch++){ // Insertion des batchs
                ArrayList<Interaction<A>> observations = total.get(batch);
                int numberObservation = observations.size() ;//Nombre données temporelles
                int numBackwards = backwardsNumber.get(batch); // Nombre de backwards parmi ces données
                for(int temporal =0 ; temporal < observations.size(); temporal++){ // INsertion temporelles
                    Interaction<A> interact = observations.get(temporal);
                    //labels globaux
                    int indice = - numberObservation +numBackwards + temporal;
                    if(indice >= 0){
                        INDArray action = (INDArray) this.learning.getActionSpace().mapActionToNumber(interact.getAction());
                        //Préparation actions
                        actions.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward), NDArrayIndex.all()}, action);
                        //Labellisation des secondes observations
                        secondObservations.put(new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.point(temporal)},interact.getSecondObservation());
                        //rewards
                        rewards.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward),NDArrayIndex.all()}, interact.getReward());
                        //Label du mask
                        maskLabel.put(new INDArrayIndex[]{NDArrayIndex.point(batch*forward + temporal),NDArrayIndex.all()},Nd4j.ones(1));
                        cursorBackward++ ;
                    }

                    //inputs observations
                    INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.point(batch),NDArrayIndex.all(),NDArrayIndex.point(temporal)};
                    inputs.put(indexs,interact.getObservation());

                    //mask index
                    INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(batch),NDArrayIndex.point(temporal)};
                    masks.put(indexMask, Nd4j.ones(1));

                }
            }

            // Apprentissage : besoin de l'état
            INDArray state_label = this.observationApproximator.forwardLearn(inputs, null, totalBatchs,masks,maskLabel);

            //Commencement de l'apprentissage, labellisation
            this.targetObservationApproximator.setMaskLabel(maskLabel);
            INDArray labels = this.labelize(secondObservations,rewards); // A faire après le forard learn pour avoir la bonne mémoire

            //Apprentissage critic
            INDArray inputCritics = Nd4j.concat(1, state_label, actions);
            INDArray epsilon = this.learn_critic(inputCritics, labels, totalBatchs*forward);
            INDArray score = this.criticApproximator.getScoreArray();
            this.experienceReplay.setError(score, backwardsNumber, backward, total);


            //Apprentissage politique
            int sizeObservation = state_label.size(1);
            int sizeAction = this.learning.getActionSpace().getSize();
            this.learn_actor(state_label, sizeObservation, sizeAction, totalBatchs*forward); // Important entre la propagation de l'observation et la backpropagation du gradient

            //Apprentissage des observations
            INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, sizeObservation));
            //INDArray labelObservation = Nd4j.zeros(numRows*forward,this.observationApproximator.numOutput());
            /*for(int k = 0 ; k < forwardsNumbers.size(); k++){
                INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(forwardsNumbers.get(k)-1)};
                labelObservation.put(indexs,epsilonObservation);
            }*/
            this.learn_observator(inputs, epsilonObservation, totalBatchs*forward, actions, state_label, labels);
            this.cpt_time++;
        }
    }

}
