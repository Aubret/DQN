package fr.univlyon1.actorcritic;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.LSTM;
import fr.univlyon1.networks.Mlp;
import org.deeplearning4j.nn.conf.Updater;
import org.nd4j.linalg.activations.Activation;

public class LstmActorCritic<A> extends ContinuousActorCritic<A> {

    public LSTM observationApproximator ;

    public LstmActorCritic(ObservationSpace observationSpace, ActionSpace<A> actionSpace, Configuration conf, long seed) {
        super(observationSpace, actionSpace, conf, seed);
    }

    public void initLstm(){
        this.observationApproximator = new LSTM(observationSpace.getShape()[0], 100, seed);
        this.observationApproximator.setLearning_rate(conf.getLearning_rateLstm());
        this.observationApproximator.setListener(true);
        this.observationApproximator.setNumNodesPerLayer(conf.getLayersLstmHiddenNodes());
        this.observationApproximator.setNumLayers(conf.getNumLstmlayers());
        this.observationApproximator.setUpdater(Updater.ADAM);
        this.observationApproximator.setEpsilon(true);

        this.observationApproximator.init() ;
    }
}
