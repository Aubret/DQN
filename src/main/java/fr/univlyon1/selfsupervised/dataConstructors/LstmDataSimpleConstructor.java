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

/**
 * Simple constuctor : Ici pour un temps constant, on veut prédire la prochaine observation
 * @param <A>
 */
public class LstmDataSimpleConstructor<A> extends LstmDataConstructor<A> {
    public LstmDataSimpleConstructor(LSTM lstm, SequentialExperienceReplay<A> timeEp, SpecificObservationReplay<A> labelEp, Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2) {
        super(lstm, timeEp, labelEp, conf, actionSpace, observationSpace, conf2);
    }

    public DataTarget choosePrediction(ArrayList<Interaction<A>> observations, int dt ) {
        if(this.configuration2.getDataBuilder().equals("DataReward")){
            return this.dataBuilder.build(null,observations.get(observations.size()-1),null);
        }
        Interaction<A> last = observations.get(observations.size()-1);
        this.labelEp.setRepere(last);
        SpecificObservation spo = this.labelEp.subset().first();

        boolean found = false ;
        int cursor = observations.size()-1 ;
        while(cursor >= 0){
            Interaction<A> inter = observations.get(cursor);
            if(inter.getIdObserver() == spo.getId()) {
                found = true;
                break ;
            }
            cursor -- ;
        }

        if(!found) {
            //System.out.println("BIZARRE NON PRESENT SIMPLE CONSTRUCTOR");
            return null;
        }

        Double tNorm = configuration.getForwardTime()/2.;
        return this.dataBuilder.build(spo,last,(spo.getOrderedNumber()-last.getTime()- tNorm)/tNorm);
    }
}
