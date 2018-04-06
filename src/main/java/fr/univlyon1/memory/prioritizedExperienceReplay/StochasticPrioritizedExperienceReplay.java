package fr.univlyon1.memory.prioritizedExperienceReplay;

import fr.univlyon1.environment.Interaction;
import fr.univlyon1.memory.ExperienceReplay;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeSet;

public class StochasticPrioritizedExperienceReplay<A> extends ExperienceReplay<A> {
    TreeSet<InteractionHistory<A>> history;
    HashMap<Interaction<A>,InteractionHistory<A>> interactions ;
    ArrayList<InteractionHistory<A>> tmp ;
    private Random random;
    private Double sum ;

    public StochasticPrioritizedExperienceReplay(int maxSize,long seed) {
        super(maxSize);
        this.resetMemory();
        this.random = new Random(seed);
        this.sum = 0. ;

    }
    //private

    @Override
    public void addInteraction(Interaction<A> interaction) {
        if(this.interactions.size() == this.maxSize) {
            InteractionHistory ih = history.pollFirst();
            this.sum -= ih.getErrorValue() ;
            this.interactions.remove(ih.getInteraction());
        }
        double val = history.size() > 0 ? history.last().getErrorValue()+1. : 1. ;
        InteractionHistory<A> newIh = new InteractionHistory<A>(interaction,val);
        this.sum+=newIh.getErrorValue() ; // important avant d'ajouter dans history
        this.history.add(newIh);
        this.interactions.put(interaction, newIh);
    }

    @Override
    public void setError(INDArray errors) {

    }

    @Override
    public Interaction<A> chooseInteraction() {
        if(this.interactions.size() > 0) {
            Double min = this.history.first().getErrorValue();
            Double scale = this.history.last().getErrorValue() -min;
            Double rand = random.nextDouble() ;
            Double toFind = (rand * scale) + min ;
            InteractionHistory<A> ih = history.higher(new InteractionHistory<A>(null,toFind));
            Interaction<A> i = ih.getInteraction();
            this.tmp.add(ih);
            return i ;
        }
        return null;
    }

    @Override
    public void resetMemory() {
        this.history = new TreeSet<InteractionHistory<A>>(new InteractionComparator<InteractionHistory>());
        this.interactions = new HashMap<>();
        this.tmp = new ArrayList<>();
    }

    @Override
    public int getSize() {
        return interactions.size();
    }

    public Double getSum() {
        return sum;
    }
}
