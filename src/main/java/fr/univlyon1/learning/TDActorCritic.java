package fr.univlyon1.learning;

import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.Interaction;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class TDActorCritic<A> extends TDBatch<A> {

    private Approximator targetActorApproximator ;
    private Approximator criticApproximator ;
    private Approximator targetCriticApproximator ;

    public TDActorCritic(double gamma, Learning<A> learning, ExperienceReplay<A> experienceReplay, int batchSize, int iterations, Approximator criticApproximator) {
        super(gamma, learning, experienceReplay, batchSize, iterations);
        this.targetActorApproximator = this.learning.getApproximator().clone(false);
        this.criticApproximator = criticApproximator ;
        this.approximator = criticApproximator;
        this.targetCriticApproximator = criticApproximator.clone(false);
    }

    /**
     * Extract mini batch
     */
    @Override
    protected void learn(){
        int numRows = Math.min(this.experienceReplay.getMemory().size(),this.batchSize);
        if(numRows <1 )
            return ;
        int sizeAction = this.learning.getActionSpace().getSize() ;
        int sizeObservation = this.lastInteraction.getObservation().size(1);
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
                labels.putRow(i, this.labelize(interaction, this.approximator));
            }
            INDArray epsilonObsAct = (INDArray)this.criticApproximator.learn(inputs, labels, numRows); // Critic learning
            INDArray epsilonAction = epsilonObsAct.get(NDArrayIndex.all(),NDArrayIndex.interval(sizeObservation,numColumns));
            this.learning.getApproximator().learn(observations,epsilonAction,numRows); //Policy learning
            //System.out.println(grad.gradientForVariable());
            //INDArray gradAction = grad.gradient().get(NDArrayIndex.point(0),NDArrayIndex.interval(sizeObservation,numColumns));
            //System.out.println(gradAction);
        }
    }

    /**
     * Renvoie les labels avec la formue r + lamba*maxQ
     * @param approximator
     */
    @Override
    protected INDArray labelize(Interaction<A> interaction, Approximator approximator){
        //INDArray entry = Nd4j.concat(1,interaction.getObservation(), (INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction())) ;
        // INDArray entryTarget = Nd4j.concat(1,interaction.getObservation(), (INDArray)this.learning.getActionSpace().mapActionToNumber(interaction.getAction())) ;
        INDArray actionTarget = this.targetActorApproximator.getOneResult(interaction.getSecondObservation());
        INDArray entryCriticTarget = Nd4j.concat(1,interaction.getSecondObservation(), actionTarget) ;
        INDArray res = this.targetCriticApproximator.getOneResult(entryCriticTarget).muli(this.gamma).addi(interaction.getReward()) ;
        return res ;
    }

    public void epoch(){
        this.targetCriticApproximator = this.criticApproximator.clone(false);
        this.targetActorApproximator = this.learning.getApproximator().clone(false);
    }

}
