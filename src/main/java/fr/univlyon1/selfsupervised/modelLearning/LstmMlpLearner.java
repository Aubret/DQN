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
import fr.univlyon1.selfsupervised.dataConstructors.LstmDataConstructor;
import fr.univlyon1.selfsupervised.dataConstructors.LstmDataNumberConstructor;
import fr.univlyon1.selfsupervised.dataConstructors.LstmDataSimpleConstructor;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;

public class LstmMlpLearner<A> extends Learner<A> {
    protected LSTM2D lstm ;
    protected LstmDataConstructor<A> dataConstructors ;
    protected int cpt ;
    protected int schedule ;
    protected Mlp regression ;


    protected double cumulScoreUp = 0.;
    protected double cumulObservation = 0.;



    public LstmMlpLearner(Approximator lstm, SupervisedConfiguration supervisedConfiguration, ExperienceReplay<A> timeEp, ExperienceReplay<A> labelEp, Configuration configuration, ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed ){
        super(supervisedConfiguration,configuration,actionSpace,observationSpace,seed);
        assert(lstm instanceof LSTM2D);
        assert(timeEp instanceof SequentialExperienceReplay);
        assert(labelEp instanceof SpecificObservationReplay);
        this.lstm =(LSTM2D)lstm;
        //this.dataConstructors = new LstmDataConstructor<A>((SequentialExperienceReplay<A>) timeEp, (SpecificObservationReplay<A>) labelEp, configuration, actionSpace, observationSpace, supervisedConfiguration);
        //this.dataConstructors = new LstmDataNumberConstructor<A>(this.lstm,(SequentialExperienceReplay<A>) timeEp, (SpecificObservationReplay<A>) labelEp, configuration, actionSpace, observationSpace, supervisedConfiguration,supervisedConfiguration.getTimeDifficulty());
        this.dataConstructors = new LstmDataSimpleConstructor<A>(this.lstm,(SequentialExperienceReplay<A>) timeEp, (SpecificObservationReplay<A>) labelEp, configuration, actionSpace, observationSpace, supervisedConfiguration);
        //this.dataConstructors = new LstmDataTimeConstructor<A>(this.lstm,(SequentialExperienceReplay<A>) timeEp, (SpecificObservationReplay<A>) labelEp, configuration, actionSpace, observationSpace, supervisedConfiguration);
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

        INDArray firstval = this.regression.getValues().detach();
        double meanScore =0.;
        if(cpt % schedule == 0){
            System.out.println( firstval.get(NDArrayIndex.point(0),NDArrayIndex.all()) + " vs " +mbd.getLabels().get(NDArrayIndex.point(0),NDArrayIndex.all()));
            System.out.println("Lstm output : "+result.get(NDArrayIndex.point(0),NDArrayIndex.all()));
            INDArray s1 = firstval.sub(mbd.getLabels());
            Double val1 = s1.muli(s1).meanNumber().doubleValue();

            INDArray newVal = this.regression.getOneResult(Nd4j.concat(1,result,mbd.getAddings()));
            INDArray s2 = newVal.sub(mbd.getLabels());
            Double val2 =s2.muli(s2).meanNumber().doubleValue();
            meanScore = val1-val2 ;
            cumulScoreUp+=meanScore ;
            System.out.println(meanScore + " -- " +cumulScoreUp);
        }
        this.lstm.learn(mbd.getInputs(), epsilonObservation, mbd.getTotalForward() * mbd.getTotalbatchs());

        if(cpt % schedule == 0) {

            INDArray s1 = firstval.sub(mbd.getLabels());
            Double val1 = s1.muli(s1).meanNumber().doubleValue();

            INDArray res = this.lstm.getOneTrainingResult(mbd.getInputs());
            INDArray inputAction = Nd4j.concat(1, res, mbd.getAddings());
            INDArray newVal = this.regression.getOneResult(inputAction);
            INDArray s2 = newVal.sub(mbd.getLabels());
            Double val2 = s2.muli(s2).meanNumber().doubleValue();
            double meanScore2 = val1 - val2;
            cumulObservation += meanScore2;
            //cumulOnlyObservation+=(val3-val2);
            System.out.println((meanScore2-meanScore) + " -- " + (cumulObservation - cumulScoreUp));
            System.out.println("------");

        }
        cpt++ ;

    }

    @Override
    public void stop() {
        System.out.println("regression stop");
        this.regression.stop();
        System.out.println("lstm saved");
        this.lstm.stop();
    }

    public void initRegression(){
        //this.policyApproximator =new Mlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0],this.actionSpace.getSize(),this.seed);
        this.regression =new Mlp(lstm.getOutput()+this.dataConstructors.getDataBuilder().getNumAddings(),this.dataConstructors.getDataBuilder().getNumPredicts(),this.seed);
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
        this.regression.setName("Regression");
        //this.regression.setL2(0.001);
        /*((NormalizedMlp)this.policyApproximator).setLayerNormalization(true);
        ((NormalizedMlp)this.policyApproximator).setBatchNormalization(false);*/

        this.regression.init() ; // A la fin
    }
}
