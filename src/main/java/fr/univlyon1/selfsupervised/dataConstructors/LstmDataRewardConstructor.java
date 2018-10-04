package fr.univlyon1.selfsupervised.dataConstructors;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.LSTM;
import fr.univlyon1.selfsupervised.dataTransfer.DataBuilder;
import fr.univlyon1.selfsupervised.dataTransfer.DataTarget;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.SortedSet;

public class LstmDataRewardConstructor<A> extends LstmDataConstructors<A> {
    public LstmDataRewardConstructor(LSTM lstm , SequentialExperienceReplay<A> timeEp, SpecificObservationReplay<A> labelEp, Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2) {
        super(lstm,timeEp, labelEp, conf, actionSpace, observationSpace, conf2);
        assert(conf2.getDataBuilder().equals("DataReward"));
        this.dataBuilder = new DataBuilder<A>(conf2.getDataBuilder(),this);
    }
    public DataTarget choosePrediction(ArrayList<Interaction<A>> observations, int dt ) {
        return this.dataBuilder.build(null,observations.get(observations.size()-1),null);
    }


}
