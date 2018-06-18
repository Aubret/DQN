package fr.univlyon1.memory.prioritizedExperienceReplay;

import fr.univlyon1.environment.Interaction;

import java.util.ArrayList;
import java.util.LinkedList;

public class StochasticForPrioritized<A> extends StochasticPrioritizedExperienceReplay<A> {
    protected SequentialPrioritizedExperienceReplay<A> seq ;
    public StochasticForPrioritized(int maxSize, long seed, ArrayList<String> file,SequentialPrioritizedExperienceReplay<A> seq) {
        super(maxSize, seed, file);
        this.seq = seq ;
        this.toTake = new LinkedList<>();
    }

    @Override
    public Interaction<A> chooseInteraction() {
        if(this.toTake.size() != 0 && this.seq.isAvailable(this.toTake.get(0).getInteraction())){
            InteractionHistory<A> ih = this.toTake.remove(0);
            this.tmp.add(ih);
            return ih.getInteraction();
        }
        if(this.history.size() > 0) {
            Double max = this.history.getTotalSum();
            Double rand = random.nextDouble() ;
            Double toFind = max * rand ;
            InteractionHistory<A> ih = history.getInteractionUp(toFind);
            Interaction<A> i = ih.getInteraction();
            this.tmp.add(ih);
            return i ;
        }
        return null;
    }
}
