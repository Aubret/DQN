package fr.univlyon1.selfsupervised.modelLearning;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import fr.univlyon1.selfsupervised.dataTransfer.MultipleModelBasedData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class GradientMultipleLearner<A> extends MultipleLearner<A>{


    public GradientMultipleLearner(Approximator lstm, ExperienceReplay<A> timeEp, ExperienceReplay<A> labelEp, SupervisedConfiguration supervisedConfiguration, Configuration configuration, ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed) {
        super(lstm, timeEp, labelEp, supervisedConfiguration, configuration, actionSpace, observationSpace, seed);
    }


    @Override
    public void learn() {
        if (cpt % schedule == 0)
            System.out.println("--------------------");
        MultipleModelBasedData mbd = (MultipleModelBasedData)this.dataConstructors.get(0).construct();
        INDArray result = this.lstm.forwardLearn(mbd.getInputs(), Nd4j.zeros(1, this.lstm.numOutput()), mbd.getTotalbatchs(), mbd.getMask(), mbd.getMaskLabel());

        INDArray epsilonObservation = Nd4j.zeros(this.lstm.numOutput());
        for(int j =0 ; j < this.regressions.size() ; j++) {
            INDArray epsilon = (INDArray) this.regressions.get(j).learn(Nd4j.concat(1, result, mbd.getAddings2().get(j)), mbd.getLabels2().get(j), mbd.getTotalbatchs());
            epsilonObservation.addi(epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.lstm.numOutput())));
            //System.out.println(epsilonObservation);
            INDArray firstval = this.regressions.get(j).getValues().detach();
            if (cpt % schedule == 0) {
                System.out.println(j+ " - "+ firstval.get(NDArrayIndex.point(0), NDArrayIndex.all()) + " vs " + mbd.getLabels2().get(j).get(NDArrayIndex.point(0), NDArrayIndex.all()));
                //System.out.println("Lstm output : " + result.get(NDArrayIndex.point(0), NDArrayIndex.all()));
            }
        }
        epsilonObservation.divi(this.regressions.size());
        this.lstm.learn(mbd.getInputs(), epsilonObservation, mbd.getTotalForward() * mbd.getTotalbatchs());

        cpt++ ;
    }


}
