package fr.univlyon1.memory.prioritizedExperienceReplay;

import fr.univlyon1.environment.Interaction;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.sumTree.SumTree;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Sum;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeSet;

public class StochasticPrioritizedExperienceReplay<A> extends ExperienceReplay<A> {
    SumTree<A> history;
    HashMap<Interaction<A>,InteractionHistory<A>> interactions ;
    ArrayList<InteractionHistory<A>> tmp ;
    private Random random;
    ArrayList<InteractionHistory<A>> toTake ;

    public StochasticPrioritizedExperienceReplay(int maxSize,long seed,ArrayList<String> file) {
        super(maxSize,file);
        this.resetMemory();
        this.random = new Random(seed);
    }
    //private

    @Override
    public void addInteraction(Interaction<A> interaction) {
        if(this.history.size() == this.maxSize) {
            InteractionHistory ih = history.getFirst();
            this.interactions.remove(ih.getInteraction());
        }
        InteractionHistory<A> newIh = new InteractionHistory<A>(interaction,1.);
        toTake.add(newIh);
        this.interactions.put(interaction, newIh);
    }

    @Override
    public void setError(INDArray errors) {
        for(int i = 0;i< this.tmp.size(); i++){
            InteractionHistory<A> ih = this.tmp.get(i);
            double error = errors.getDouble(i);
            ih.computeError(error); // Important de le faire avant
            this.history.insert(ih);
        }
        this.tmp = new ArrayList<>();
    }

    @Override
    public Interaction<A> chooseInteraction() {
        if(this.random.nextBoolean() || this.history.size() == 0) {
            if (this.toTake.size() != 0) {
                InteractionHistory<A> ih = this.toTake.remove(toTake.size() - 1);
                //System.out.println(toTake.size());
                this.tmp.add(ih);
                return ih.getInteraction();
            }
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

    @Override
    public void resetMemory() {
        this.history = new SumTree<A>();
        this.interactions = new HashMap<>();
        this.tmp = new ArrayList<>();
        this.toTake = new ArrayList<>();
    }

    @Override
    public int getSize() {
        return this.history.size()+this.toTake.size();
    }


}
