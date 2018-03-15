package main.java.fr.univlyon1.learning;

import main.java.fr.univlyon1.actorcritic.Learning;
import main.java.fr.univlyon1.environment.Interaction;
import main.java.fr.univlyon1.memory.ExperienceReplay;
import main.java.fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TDBatch<A> extends TD<A> {
    private ExperienceReplay<A> experienceReplay ;
    private int batchSize ;
    private int nbrIterations;
    private Approximator approximator ;

    public TDBatch(double gamma, Learning<A> learning,ExperienceReplay<A> experienceReplay, int batchSize, int iterations) {
        super(gamma, learning);
        this.experienceReplay = experienceReplay;
        this.batchSize= batchSize ;
        this.nbrIterations = iterations ;
        this.approximator = learning.getApproximator().clone(true);
    }

    @Override
    public void evaluate(INDArray input, Double reward) {
        if(this.lastInteraction != null) { // Avoir des interactions complètes
            this.lastInteraction.setSecondObservation(input);
            this.lastInteraction.setReward(reward);
            this.experienceReplay.addInteraction(this.lastInteraction.clone());
            this.learn();
        }
    }


    @Override
    protected void learn(){
        int numRows = Math.min(this.experienceReplay.getMemory().size(),this.batchSize);
        if(numRows <1 )
            return ;
        int numColumns = this.lastInteraction.getObservation().size(1);
        int numColumnsLabels = this.learning.getActionSpace().getSize();
        for(int j = 0;j<this.nbrIterations;j++) {
            INDArray inputs = Nd4j.zeros(numRows, numColumns);
            INDArray labels = Nd4j.zeros(numRows, numColumnsLabels);
            for (int i = 0; i < numRows; i++) {
                Interaction<A> interaction = experienceReplay.chooseInteraction();
                inputs.putRow(i, interaction.getObservation());
                labels.putRow(i, this.labelize(interaction, this.approximator));
            }
            this.approximator.learn(inputs, labels, numRows);
        }
    }
}
