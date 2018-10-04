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
import fr.univlyon1.networks.NormalizedMlp;
import fr.univlyon1.networks.lossFunctions.LossError;
import fr.univlyon1.networks.lossFunctions.LossMseSaveScore;
import fr.univlyon1.selfsupervised.dataConstructors.LstmDataConstructors;
import fr.univlyon1.selfsupervised.dataConstructors.LstmDataNumberConstructor;
import fr.univlyon1.selfsupervised.dataConstructors.LstmDataRewardConstructor;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.IOException;

public class LstmMlpLearner<A> implements Learner {
    protected LSTM2D lstm;
    protected Mlp regression ;
    protected SupervisedConfiguration conf;
    protected LstmDataConstructors<A> dataConstructors ;
    protected ActionSpace<A> actionSpace ;
    protected long seed ;
    protected int cpt ;
    protected int schedule ;

    protected double cumulScoreUp = 0.;


    public LstmMlpLearner(Approximator lstm, SupervisedConfiguration supervisedConfiguration, ExperienceReplay<A> timeEp, ExperienceReplay<A> labelEp, Configuration configuration, ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed ){
        assert(lstm instanceof LSTM2D);
        assert(timeEp instanceof SequentialExperienceReplay);
        assert(labelEp instanceof SpecificObservationReplay);
        this.lstm =(LSTM2D)lstm;
        this.conf = supervisedConfiguration ;
        this.actionSpace = actionSpace ;
        if(supervisedConfiguration.getDataBuilder().equals("DataReward"))
            this.dataConstructors = new LstmDataRewardConstructor<A>(this.lstm,(SequentialExperienceReplay<A>)timeEp,(SpecificObservationReplay<A>)labelEp,configuration,actionSpace,observationSpace,supervisedConfiguration);
        else {
            //this.dataConstructors = new LstmDataConstructors<A>((SequentialExperienceReplay<A>) timeEp, (SpecificObservationReplay<A>) labelEp, configuration, actionSpace, observationSpace, supervisedConfiguration);
            this.dataConstructors = new LstmDataNumberConstructor<A>(this.lstm,(SequentialExperienceReplay<A>) timeEp, (SpecificObservationReplay<A>) labelEp, configuration, actionSpace, observationSpace, supervisedConfiguration);
        }
        this.seed=seed ;
        this.schedule = 200 ;
        this.cpt= 0 ;
        this.initRegression();
    }

    public void learn(){
        ModelBasedData mbd = this.dataConstructors.construct();
        INDArray result =this.lstm.forwardLearn(mbd.getInputs(), Nd4j.zeros(1,this.lstm.numOutput()),mbd.getTotalbatchs(),mbd.getMask(),mbd.getMaskLabel());
        INDArray epsilon = (INDArray)this.regression.learn(Nd4j.concat(1,result,mbd.getAddings()),mbd.getLabels(),mbd.getTotalbatchs());
        INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.lstm.numOutput()));
        //System.out.println(epsilonObservation);
        this.lstm.learn(mbd.getInputs(),epsilonObservation,mbd.getTotalForward()*mbd.getTotalbatchs());

        if(cpt % schedule == 0){
            INDArray firstval = this.regression.getValues().detach();
            System.out.println( firstval.get(NDArrayIndex.point(0),NDArrayIndex.all()) + " vs " +mbd.getLabels().get(NDArrayIndex.point(0),NDArrayIndex.all()));
            System.out.println("Lstm output : "+result.get(NDArrayIndex.point(0),NDArrayIndex.all()));
            INDArray s1 = firstval.sub(mbd.getLabels());
            Double val1 = s1.muli(s1).meanNumber().doubleValue();

            INDArray inputs = this.lstm.getOneTrainingResult(mbd.getInputs());
            INDArray newVal = this.regression.getOneResult(Nd4j.concat(1,inputs,mbd.getAddings()));
            INDArray s2 = newVal.sub(mbd.getLabels());
            Double val2 =s2.muli(s2).meanNumber().doubleValue();
            double meanScore = val1-val2 ;
            cumulScoreUp+=meanScore ;
            System.out.println(meanScore + " -- " +cumulScoreUp);
            System.out.println("------");
        }
        cpt++ ;

    }

    @Override
    public void stop() {
        System.out.println("regression stop");
        this.regression.stop();
        System.out.println("lstm saved");

    }

    public void initRegression(){
        //this.policyApproximator =new Mlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0],this.actionSpace.getSize(),this.seed);
        this.regression =new Mlp(lstm.getOutput()+this.dataConstructors.numAddings(),this.dataConstructors.getDataBuilder().getNumPredicts(),this.seed);
        this.regression.setLearning_rate(conf.getLearning_rate());
        this.regression.setNumNodes(conf.getNumHiddenNodes());
        this.regression.setNumLayers(conf.getNumLayers());
        //this.policyApproximator.setEpsilon(true); // On retropropage le gradient avec epsilon et non une foncitno de perte
        this.regression.setLossFunction(new LossMseSaveScore());
        this.regression.setEpsilon(false);
        this.regression.setMinimize(true); // On souhaite minimiser le gradient
        this.regression.setListener(true);
        this.regression.setUpdater(new Adam(conf.getLearning_rate()));
        this.regression.setLastActivation(Activation.IDENTITY);
        this.regression.setHiddenActivation(Activation.ELU);
        this.regression.setNumNodesPerLayer(conf.getLayersHiddenNodes());
        //this.regression.setL2(0.001);
        /*((NormalizedMlp)this.policyApproximator).setLayerNormalization(true);
        ((NormalizedMlp)this.policyApproximator).setBatchNormalization(false);*/

        this.regression.init() ; // A la fin
    }
}
