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
import fr.univlyon1.selfsupervised.dataConstructors.MultipleLstmDataNumberConstructor;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import fr.univlyon1.selfsupervised.dataTransfer.MultipleModelBasedData;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;

import java.util.ArrayList;

public class GradientMultipleLearner<A> extends Learner<A>{
    protected DataConstructor<A> dataConstructor ;
    protected int cpt ;
    protected int schedule ;
    protected ArrayList<Mlp> regressions ;
    protected LSTM2D lstm ;
    protected SequentialExperienceReplay<A> timeEp ;
    protected SpecificObservationReplay<A> labelEp ;

    public GradientMultipleLearner(Approximator lstm, ExperienceReplay<A> timeEp, ExperienceReplay<A> labelEp, SupervisedConfiguration supervisedConfiguration, Configuration configuration, ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed) {
        super(supervisedConfiguration, configuration, actionSpace, observationSpace, seed);
        this.lstm = (LSTM2D)lstm ;
        this.timeEp = (SequentialExperienceReplay<A>)timeEp ;
        this.labelEp = (SpecificObservationReplay<A>)labelEp ;
        this.cpt = 0 ;
        this.schedule = 200 ;
        this.regressions = new ArrayList<>();
        ArrayList<Integer> dts = new ArrayList<>();
        for(int i = 0 ; i < this.conf.getTimeDifficulty() ; i++) {
            dts.add(i) ;
        }
        this.dataConstructor = new MultipleLstmDataNumberConstructor<A>(this.timeEp,this.labelEp, configuration, actionSpace, observationSpace, this.conf,dts);
        for(int i = 0; i < this.conf.getTimeDifficulty(); i++){
            this.regressions.add(this.initRegression(this.dataConstructor,i));
        }
    }

    @Override
    public void learn() {
        if (cpt % schedule == 0)
            System.out.println("--------------------");
        MultipleModelBasedData mbd = (MultipleModelBasedData)this.dataConstructor.construct();
        INDArray result = this.lstm.forwardLearn(mbd.getInputs(), Nd4j.zeros(1, this.lstm.numOutput()), mbd.getTotalbatchs(), mbd.getMask(), mbd.getMaskLabel());

        INDArray epsilonObservation = Nd4j.zeros(mbd.getTotalbatchs(),this.lstm.numOutput());
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
