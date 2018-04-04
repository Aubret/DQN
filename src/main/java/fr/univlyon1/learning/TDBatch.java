package fr.univlyon1.learning;

import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.actorcritic.Learning;
import fr.univlyon1.environment.Interaction;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TDBatch<A> extends TD<A> {
    protected ExperienceReplay<A> experienceReplay ;
    protected int batchSize ;
    protected int nbrIterations;
    protected Approximator approximator ;

    public TDBatch(double gamma, Learning<A> learning,ExperienceReplay<A> experienceReplay, int batchSize, int iterations) {
        super(gamma, learning);
        this.experienceReplay = experienceReplay;
        this.batchSize= batchSize ;
        this.nbrIterations = iterations ;
        this.approximator = learning.getApproximator().clone(true);
    }

    @Override
    public void evaluate(INDArray input, Double reward) { // Store transistions
        if(this.lastInteraction != null) { // Avoir des interactions complètes
            this.lastInteraction.setSecondObservation(input);
            this.lastInteraction.setReward(reward);
            this.experienceReplay.addInteraction(this.lastInteraction.clone());
            this.learn();
        }
    }


    /**
     * Extract mini batch
     */
    @Override
    protected void learn(){
        int numRows = Math.min(this.experienceReplay.getSize(),this.batchSize);
        if(numRows <1 )
            return ;
        int numColumns = this.lastInteraction.getObservation().size(1);
        int numColumnsLabels = this.approximator.numOutput();
        //this.learning.getActionSpace().getSize();
        for(int j = 0;j<this.nbrIterations;j++) {
            INDArray inputs = Nd4j.zeros(numRows, numColumns);
            INDArray labels = Nd4j.zeros(numRows, numColumnsLabels);
            for (int i = 0; i < numRows; i++) {
                Interaction<A> interaction = experienceReplay.chooseInteraction();
                inputs.putRow(i, interaction.getObservation());
                labels.putRow(i, this.labelize(interaction, this.approximator));
            }
            this.learning.getApproximator().learn(inputs, labels, numRows);
        }
    }

    public void epoch(){
        this.approximator = this.learning.getApproximator().clone(false);
    }
}
