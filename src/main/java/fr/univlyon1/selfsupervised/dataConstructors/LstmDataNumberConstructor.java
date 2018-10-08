package fr.univlyon1.selfsupervised.dataConstructors;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.LSTM;

import java.util.ArrayList;

public class LstmDataNumberConstructor<A> extends LstmDataConstructor<A> {
    public LstmDataNumberConstructor(LSTM lstm,SequentialExperienceReplay<A> timeEp, SpecificObservationReplay<A> labelEp, Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2, int timeDifficulty) {
        super(lstm,timeEp, labelEp, conf, actionSpace, observationSpace, conf2);
        this.timeDifficulty = timeDifficulty;
    }


    protected Interaction<A> chooseStudiedInteraction(ArrayList<Interaction<A>> observations, int number){
        this.cursor = observations.size()-1;
        while((observations.size()-1 - cursor < this.timeDifficulty) && cursor >=0 && (observations.size() - cursor) < this.numberMax){ // Choix de véhicule qu'on prédit
            cursor -- ;
        }
        if(cursor==-1)
            return null ;
        return observations.get(cursor);
    }
}
