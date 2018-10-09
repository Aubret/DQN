package fr.univlyon1.selfsupervised.modelLearning;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.LSTM2D;
import fr.univlyon1.networks.Mlp;
import fr.univlyon1.networks.lossFunctions.LossMseSaveScore;
import fr.univlyon1.selfsupervised.dataConstructors.DataConstructor;
import fr.univlyon1.selfsupervised.dataConstructors.LstmDataNumberConstructor;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;

import java.util.ArrayList;

public class MultipleLearner<A> extends Learner<A>{
    protected LSTM2D lstm ;
    protected ArrayList<DataConstructor<A>> dataConstructors ;
    protected int cpt ;
    protected int schedule ;
    protected ArrayList<Mlp> regressions ;
    protected SequentialExperienceReplay<A> timeEp ;
    protected SpecificObservationReplay<A> labelEp ;


    public MultipleLearner(Approximator lstm, ExperienceReplay<A> timeEp, ExperienceReplay<A> labelEp, SupervisedConfiguration supervisedConfiguration, Configuration configuration, ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed) {
        super(supervisedConfiguration, configuration, actionSpace, observationSpace, seed);
        assert(lstm instanceof LSTM2D);
        assert(timeEp instanceof SequentialExperienceReplay);
        assert(labelEp instanceof SpecificObservationReplay);
        this.regressions = new ArrayList<>();
        this.dataConstructors = new ArrayList<>();
        this.cpt = 0 ;
        this.schedule = 200 ;
        this.lstm = (LSTM2D) lstm ;
        this.timeEp = (SequentialExperienceReplay<A>)timeEp ;
        this.labelEp = (SpecificObservationReplay<A>)labelEp ;

        //DataConstructor<A> dc = new LstmDataNumberConstructor<A>(this.lstm, (SequentialExperienceReplay<A>) timeEp, (SpecificObservationReplay<A>) labelEp, configuration, actionSpace, observationSpace, supervisedConfiguration,4);
        //this.dataConstructors.add(dc);
        //this.regressions.add(this.initRegression(dc));
        for(int i = 0 ; i < this.conf.getTimeDifficulty() ; i++) {
            int k=i*2 ;
            DataConstructor<A> dc = new LstmDataNumberConstructor<A>(this.lstm, this.timeEp,this.labelEp, configuration, actionSpace, observationSpace, this.conf,k);
            this.dataConstructors.add(dc);
            this.regressions.add(this.initRegression(dc,k));
        }

    }

    @Override
    public void learn() {
        if (cpt % schedule == 0)
            System.out.println("--------------------");

        for(int j =0 ; j < this.regressions.size() ; j++) {
            ModelBasedData mbd = this.dataConstructors.get(j).construct();
            INDArray result = this.lstm.forwardLearn(mbd.getInputs(), Nd4j.zeros(1, this.lstm.numOutput()), mbd.getTotalbatchs(), mbd.getMask(), mbd.getMaskLabel());

            INDArray epsilon = (INDArray) this.regressions.get(j).learn(Nd4j.concat(1, result, mbd.getAddings()), mbd.getLabels(), mbd.getTotalbatchs());
            INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.lstm.numOutput()));
            //System.out.println(epsilonObservation);
            INDArray firstval = this.regressions.get(j).getValues().detach();
            if (cpt % schedule == 0) {
                System.out.println(j+ " - "+ firstval.get(NDArrayIndex.point(0), NDArrayIndex.all()) + " vs " + mbd.getLabels().get(NDArrayIndex.point(0), NDArrayIndex.all()));
                //System.out.println("Lstm output : " + result.get(NDArrayIndex.point(0), NDArrayIndex.all()));
            }
            this.lstm.learn(mbd.getInputs(), epsilonObservation, mbd.getTotalForward() * mbd.getTotalbatchs());
        }
        cpt++ ;
    }

    @Override
    public void stop() {
        System.out.println("regression stop");
        for(int i = 0 ; i < this.regressions.size() ; i++) {
            this.regressions.get(i).stop();
        }
        System.out.println("lstm saved");
        this.lstm.stop();
    }

    public Mlp initRegression(DataConstructor dataConstructor, int num){
        //this.policyApproximator =new Mlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0],this.actionSpace.getSize(),this.seed);
        Mlp mlp=new Mlp(lstm.getOutput()+dataConstructor.getDataBuilder().getNumAddings(),dataConstructor.getDataBuilder().getNumPredicts(),this.seed);
        mlp.setLearning_rate(conf.getLearning_rate());
        mlp.setNumNodes(conf.getNumHiddenNodes());
        mlp.setNumLayers(conf.getNumLayers());
        //this.policyApproximator.setEpsilon(true); // On retropropage le gradient avec epsilon et non une foncitno de perte
        mlp.setLossFunction(new LossMseSaveScore());
        mlp.setEpsilon(false);
        mlp.setMinimize(true); // On souhaite minimiser le gradient
        mlp.setListener(true);
        mlp.setUpdater(new Adam(conf.getLearning_rate()));
        mlp.setLastActivation(Activation.IDENTITY);
        mlp.setHiddenActivation(Activation.ELU);
        mlp.setNumNodesPerLayer(conf.getLayersHiddenNodes());
        mlp.setName("Regression "+num);
        //this.regression.setL2(0.001);
        /*((NormalizedMlp)this.policyApproximator).setLayerNormalization(true);
        ((NormalizedMlp)this.policyApproximator).setBatchNormalization(false);*/

        mlp.init() ; // A la fin
        return mlp ;
    }

}
