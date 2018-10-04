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
import fr.univlyon1.selfsupervised.dataTransfer.DataTarget;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.SortedSet;

public class LstmDataNumberConstructor<A> extends LstmDataConstructors<A> {
    public LstmDataNumberConstructor(LSTM lstm,SequentialExperienceReplay<A> timeEp, SpecificObservationReplay<A> labelEp, Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2) {
        super(lstm,timeEp, labelEp, conf, actionSpace, observationSpace, conf2);
    }


    protected Interaction<A> chooseStudiedInteraction(ArrayList<Interaction<A>> observations, int number){
        Double tmpT = 0. ;
        this.cursor = observations.size()-1;
        while((observations.size() - cursor < number) && cursor >=0 && (observations.size() - cursor) < this.numberMax){ // Choix de véhicule qu'on prédit
            cursor -- ;
        }
        if(cursor==-1)
            return null ;
        return observations.get(cursor);
    }
}
