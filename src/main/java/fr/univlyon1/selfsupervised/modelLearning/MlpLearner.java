package fr.univlyon1.selfsupervised.modelLearning;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Mlp;
import fr.univlyon1.networks.lossFunctions.LossMseSaveScore;
import fr.univlyon1.selfsupervised.dataConstructors.MlpDataConstructor;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;

public class MlpLearner<A> extends Learner<A>{

    protected ExperienceReplay<A> ep ;
    protected MlpDataConstructor<A> dataConstructor ;
    protected Mlp regression ;
    protected int cpt ;
    protected int schedule ;
    protected double cumulScoreUp = 0.;



    public MlpLearner( SupervisedConfiguration supervisedConfiguration, ExperienceReplay<A> ep, ExperienceReplay<A> spo, Configuration configuration, ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed) {
        super(supervisedConfiguration, configuration, actionSpace, observationSpace, seed);
        assert !(ep instanceof SequentialExperienceReplay)  : "mauvais experience replay" ;
        assert(spo instanceof SpecificObservationReplay);
        if(ep instanceof SequentialExperienceReplay){
            System.out.println("errroooor !!!!");
        }
        this.ep = ep ;
        this.dataConstructor = new MlpDataConstructor<A>(ep,(SpecificObservationReplay<A>)spo,configuration,actionSpace,observationSpace,conf);
        this.initRegression();
        this.schedule = 200 ;
        this.cpt= 0 ;
    }

    @Override
    public void learn() {
        ModelBasedData mbd = this.dataConstructor.construct();
        //this.regression.learn(Nd4j.concat(1,mbd.getInputs(),mbd.getAddings()),mbd.getLabels(),mbd.getTotalbatchs());
        this.regression.learn(mbd.getAddings(),mbd.getLabels(),mbd.getTotalbatchs());

        if(cpt % schedule == 0){
            INDArray firstval = this.regression.getValues().detach();
            INDArray s1 = firstval.sub(mbd.getLabels());
            Double val1 = s1.muli(s1).meanNumber().doubleValue();

            //INDArray newVal = this.regression.getOneResult(Nd4j.concat(1,mbd.getInputs(),mbd.getAddings()));
            INDArray newVal = this.regression.getOneResult(mbd.getAddings());

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
    }


    public void initRegression(){
        //this.policyApproximator =new Mlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0],this.actionSpace.getSize(),this.seed);
        //this.regression =new Mlp(this.dataConstructor.getDataBuilder().getNumAddings(),this.dataConstructor.getDataBuilder().getNumPredicts(),this.seed);
        this.regression =new Mlp(this.dataConstructor.getDataBuilder().getNumAddings(),this.dataConstructor.getDataBuilder().getNumPredicts(),this.seed);

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
