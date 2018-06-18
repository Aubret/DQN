package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.configurations.SavesLearning;
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
            //contruction des INDArrays
            int totalBatchs = total.size();
            backward = backward - totalBatchs;
            INDArray secondObservations = Nd4j.zeros(totalBatchs,size,forward);
            INDArray secondObservations2 = Nd4j.zeros(backward,this.learning.getObservationSpace().getShape()[0]);// On avait besoin de la taille maximale du forward

            INDArray rewards = Nd4j.zeros(backward,1);
            //INDArray gammas = Nd4j.zeros(backward,1);

            INDArray actions = Nd4j.zeros(backward,this.learning.getActionSpace().getSize());
            INDArray inputs2 = Nd4j.zeros(backward,this.learning.getObservationSpace().getShape()[0]);// On avait besoin de la taille maximale du forward

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
                        //gammas.put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward),NDArrayIndex.all()}, interact.computeGamma());

                        cursorBackward++ ;
                    }
                    if(indice >= 0 && temporal != numberObservation-1){
                        //Label du mask
                        maskLabel.put(new INDArrayIndex[]{NDArrayIndex.point(batch*forward + temporal),NDArrayIndex.all()},Nd4j.ones(1));
                        //Labellisation des secondes observations
                        secondObservations.put(new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.point(temporal)},Nd4j.concat(1,interact.getSecondObservation(),action2));
                    }

                    if(temporal != numberObservation-1) {
                        //inputs observations
                        INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.point(temporal)};
                        inputs.put(indexs, Nd4j.concat(1, interact.getObservation(), action));

                        //mask index
                        INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(batch),NDArrayIndex.point(temporal)};
                        masks.put(indexMask, Nd4j.ones(1));
                    }
                }
            }

            // Apprentissage : besoin de l'état
            INDArray state = this.observationApproximator.forwardLearn(inputs, null, totalBatchs,masks,maskLabel);
            INDArray state_label = Nd4j.concat(1,state,inputs2);
            int sizeObservation = state_label.size(1);

            //Commencement de l'apprentissage, labellisation
            this.targetObservationApproximator.setMaskLabel(maskLabel);
            INDArray obs1 = inputs.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(0));
            //INDArray labels = this.labelize(secondObservations,rewards,secondObservations2,gammas); // A faire après le forard learn pour avoir la bonne mémoire
            INDArray labels = this.labelize(secondObservations,rewards,secondObservations2,obs1); // A faire après le forard learn pour avoir la bonne mémoire

            //Apprentissage critic
            INDArray inputCritics = Nd4j.concat(1, state_label,actions);
            INDArray epsilon = this.learn_critic(inputCritics, labels, totalBatchs*forward,sizeObservation);
            this.savelearning.add(this.criticApproximator.getScore());
            INDArray score = this.criticApproximator.getScoreArray();
            this.experienceReplay.setError(score, backwardsNumber, backward, total);

            //erreur sur la politique
            //INDArray epsilonAction = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(sizeObservation, sizeObservation+this.learning.getActionSpace().getSize()));
            //INDArray epsilonActor = this.learning.getApproximator().error(state_label, epsilonAction,totalBatchs*forward);

            //Apprentissage politique
            int sizeAction = this.learning.getActionSpace().getSize();
            this.learn_actor(state_label, sizeObservation, sizeAction, totalBatchs*forward); // Important entre la propagation de l'observation et la backpropagation du gradient

            //INDArray epsilonObservationCrit = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.observationApproximator.numOutput()));
            INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.observationApproximator.numOutput()));
            //INDArray epsilonObservationAct = epsilonActor.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.observationApproximator.numOutput()));
            //INDArray epsilonObservation = epsilonObservationCrit.addi(epsilonObservationAct);
            this.learn_observator(inputs, epsilonObservation, totalBatchs*forward, actions, inputs2, labels);
            this.cpt_time++;
        }
    }

        protected INDArray labelize(INDArray secondObservations ,INDArray rewards,INDArray secondObservations2,INDArray obs1){
        // Les états précédents sont dans la mémoire de l'approximateur
        this.targetObservationApproximator.clear();
        this.targetObservationApproximator.getOneResult(obs1);

        //this.targetObservationApproximator.setMemory(this.observationApproximator.getSecondMemory());
        INDArray state = this.targetObservationApproximator.getOneResult(secondObservations);
        INDArray stateLabel = Nd4j.concat(1,state,secondObservations2);

        INDArray action = this.targetActorApproximator.getOneResult(stateLabel);
        INDArray entryCriticTarget = Nd4j.concat(1,stateLabel, action) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        res = res.muli(this.gamma);
        res.addi(rewards) ;
        return res ;
    }

    protected INDArray labelize(INDArray secondObservations ,INDArray rewards,INDArray secondObservations2/*,INDArray gammas*/){
        // Les états précédents sont dans la mémoire de l'approximateur
        //this.targetObservationApproximator.setMemory(this.observationApproximator.getSecondMemory());
        this.targetObservationApproximator.clear();


        INDArray state = this.targetObservationApproximator.getOneResult(secondObservations);
        INDArray stateLabel = Nd4j.concat(1,state,secondObservations2);

        INDArray action = this.targetActorApproximator.getOneResult(stateLabel);
        INDArray entryCriticTarget = Nd4j.concat(1,stateLabel, action) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        //res = res.muli(gammas);
        res.addi(rewards) ;
        return res ;
    }


}
