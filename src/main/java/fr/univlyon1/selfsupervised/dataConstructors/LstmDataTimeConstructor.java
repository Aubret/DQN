package fr.univlyon1.selfsupervised.dataConstructors;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.LSTM;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;

import java.util.ArrayList;

/**
 * Le nombre d'interaction avant celle que l'on souhaite prédire est dynamique
 * @param <A>
 */
public class LstmDataTimeConstructor<A> extends LstmDataConstructor<A> {
    protected int scheduleSequence = 1000;
    protected int cpt = 1 ;

    public LstmDataTimeConstructor(LSTM lstm, SequentialExperienceReplay<A> timeEp, SpecificObservationReplay<A> labelEp, Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2) {
        super(lstm, timeEp, labelEp, conf, actionSpace, observationSpace, conf2);

    }

    public ModelBasedData construct() {
        ModelBasedData mbd = super.construct() ;
        if(cpt%scheduleSequence == 0){
            this.sequenceSize += 1;
            this.srd = new UniformIntegerDistribution(5,this.sequenceSize);
            //this.timeEp.setSequenceSize(this.sequenceSize);
            //this.timeEp.setBackpropSize(this.sequenceSize);
        }
        cpt++ ;
        return mbd;
    }



    protected Interaction<A> chooseStudiedInteraction(ArrayList<Interaction<A>> observations, int number){
        this.cursor = observations.size()-1;
        while((observations.size()-1 - cursor < number) && cursor >=0 && (observations.size() - cursor) < this.numberMax){ // Choix de véhicule qu'on prédit
            cursor -- ;
        }
        if(cursor==-1)
            return null ;
        return observations.get(cursor);
    }

}
