package fr.univlyon1.learning;

import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.Interaction;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.Mlp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TDLstm<A> extends TDActorCritic<A> {

    public Approximator observationApproximator ;
    public boolean replay = false ;

    public TDLstm(double gamma, Learning<A> learning, ExperienceReplay<A> experienceReplay, int batchSize, int iterations, Approximator criticApproximator, Approximator cloneCriticApproximator, Approximator observationApproximator) {
        super(gamma, learning, experienceReplay, batchSize, iterations, criticApproximator, cloneCriticApproximator);
        this.observationApproximator = observationApproximator ;
    }

    /**
     * Extract mini batch
     */
    @Override
    public void learn(){
        if(this.replay)
            learn_replay();
        else
            learn_online();
    }

    protected void learn_online(){
        INDArray labels = this.labelize(this.lastInteraction);

        //INDArray obs = this.observationApproximator.getOneResult(this.lastInteraction.getObservation());
        //this.lastInteraction.setSecondState(obs);
        //this.learn_critic(obs,);
    }

    protected void learn_critic(INDArray inputs, INDArray labels, int numRows){
        //System.out.println("----");
        //INDArray epsilonObsAct = (INDArray)this.criticApproximator.learn(inputs, labels, numRows); // Critic learning
        INDArray scores = (INDArray) this.criticApproximator.learn(inputs, labels, numRows);// Critic learning
        this.experienceReplay.setError(scores);

        if(this.cpt_time%this.time == 0){
            System.out.println("-------------");
            INDArray firstval = ((Mlp) this.criticApproximator).getValues();
            INDArray s1 = firstval.sub(labels);
            Double val1 = s1.muli(s1).meanNumber().doubleValue();
            INDArray newVal = this.criticApproximator.getOneResult(inputs);
            INDArray s2 = newVal.sub(labels);
            Double val2 =s2.muli(s2).meanNumber().doubleValue();
            double meanScore = val1-val2 ;
            cumulScoreUp+=meanScore ;
            System.out.println(meanScore + " -- " +cumulScoreUp);
        }
    }

    protected INDArray labelize(Interaction<A> interaction){
        //INDArray entry = Nd4j.concat(1,interaction.getObservation(), (INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction())) ;
        // INDArray entryTarget = Nd4j.concat(1,interaction.getObservation(), (INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction())) ;
        INDArray actionTarget = this.targetActorApproximator.getOneResult(interaction.getSecondObservation());
        //INDArray actionTarget = Nd4j.zeros(this.learning.getActionSpace().getSize()).add(0.1);
        INDArray entryCriticTarget = Nd4j.concat(1,interaction.getSecondObservation(), actionTarget) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget);
        res = res.muli(this.gamma).addi(interaction.getReward()) ;
        return res ;
    }






    protected void learn_replay(){
        int numRows = Math.min(this.experienceReplay.getSize(),this.batchSize);
        int sizeObservation ;
        if(this.lastInteraction != null && this.lastInteraction.getSecondObservation() != null ) {
            INDArray inputAction1 = Nd4j.concat(1, this.lastInteraction.getObservation(), (INDArray) this.learning.getActionSpace().mapActionToNumber(this.lastInteraction.getAction()));
            this.scoreI = this.criticApproximator.getOneResult(inputAction1).getDouble(0) - this.labelize(this.lastInteraction).getDouble(0);
            sizeObservation = this.lastInteraction.getObservation().size(1);
        }else{
            sizeObservation = this.learning.getObservationSpace().getShape()[0];
        }
        if(numRows <1 )
            return ;
        int sizeAction = this.learning.getActionSpace().getSize() ;
        int numColumns =sizeObservation+sizeAction;
        int numColumnsLabels = this.targetCriticApproximator.numOutput();
        //this.learning.getActionSpace().getSize();
        for(int j = 0;j<this.nbrIterations;j++) {
            INDArray observations = Nd4j.zeros(numRows, sizeObservation);
            INDArray inputs = Nd4j.zeros(numRows, numColumns);
            INDArray labels = Nd4j.zeros(numRows, numColumnsLabels);
            for (int i = 0; i < numRows; i++) {
                Interaction<A> interaction = experienceReplay.chooseInteraction();
                INDArray inputAction = Nd4j.concat(1,interaction.getObservation(),(INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction()));
                inputs.putRow(i, inputAction);
                observations.putRow(i, interaction.getObservation());
                labels.putRow(i, this.labelize(interaction));
            }
            this.learn_critic(inputs,labels,numRows);
            this.cloneCriticApproximator.setParams(this.criticApproximator.getParams()); // Dupliquer les paramètres
            this.learn_actor(observations,sizeObservation,numColumns,numRows);
            this.cpt_time++ ;

        }
    }
}
