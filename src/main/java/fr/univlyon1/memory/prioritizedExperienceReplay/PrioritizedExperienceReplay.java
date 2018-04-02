package fr.univlyon1.memory.prioritizedExperienceReplay;

import fr.univlyon1.environment.Interaction;
import fr.univlyon1.memory.ExperienceReplay;

import java.util.*;

/**
 * TODO
 * @param <A>
 */
public class PrioritizedExperienceReplay<A> extends ExperienceReplay<A> {
    TreeSet<InteractionHistory<A>> history;
    HashMap<Interaction<A>,InteractionHistory<A>> interactions ;
    ArrayList<InteractionHistory> tmp ;
    //private

    public PrioritizedExperienceReplay(int maxSize) {
        super(maxSize);
        this.resetMemory();
    }

    @Override
    public void addInteraction(Interaction<A> interaction) {
        if(this.interactions.size() == this.maxSize) {
            history.pollFirst();
        }
        InteractionHistory<A> newIh = new InteractionHistory<A>(interaction,1.);
        this.history.add(newIh);
        this.interactions.put(interaction, newIh);
    }

    public void setError(Interaction i, double error){
        //this.tmp.remove(ih);
        //this.history.get(interaction).computeError(error);
    }

    @Override
    public Interaction<A> chooseInteraction() {
        if(this.interactions.size() > 0) {
            InteractionHistory<A> ih = this.history.pollLast();
            Interaction<A> i = ih.getInteraction();
            this.tmp.add(ih);
            return i ;
        }
        return null;
    }

    @Override
    public List<Interaction<A>> getMemory() {
        return this.memory;
    }

    @Override
    public void resetMemory() {
        this.history = new TreeSet<InteractionHistory<A>>(new InteractionComparator<InteractionHistory>());
        this.interactions = new HashMap<>();
        this.tmp = new ArrayList<>();
    }

    @Override
    public int getSize() {
        return 0;
    }
}
