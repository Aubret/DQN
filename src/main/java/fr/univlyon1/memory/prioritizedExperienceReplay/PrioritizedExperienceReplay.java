package fr.univlyon1.memory.prioritizedExperienceReplay;

import fr.univlyon1.environment.Interaction;
import fr.univlyon1.memory.ExperienceReplay;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 *
 * @param <A>
 */
public class PrioritizedExperienceReplay<A> extends ExperienceReplay<A> {
    TreeSet<InteractionHistory<A>> history;
    HashMap<Interaction<A>,InteractionHistory<A>> interactions ;
    ArrayList<InteractionHistory<A>> tmp ;
    //private

    public PrioritizedExperienceReplay(int maxSize,String file) {
        super(maxSize,file);
        this.resetMemory();
    }

    @Override
    public void addInteraction(Interaction<A> interaction) {
        if(this.interactions.size() == this.maxSize) {
            InteractionHistory ih = history.pollFirst();
            this.interactions.remove(ih.getInteraction());
        }
        double val = history.size() > 0 ? history.last().getErrorValue()+1. : 1. ;
        InteractionHistory<A> newIh = new InteractionHistory<A>(interaction,val);
        this.history.add(newIh);
        this.interactions.put(interaction, newIh);
    }

    public void setError(INDArray errors){
        for(int i = 0;i< this.tmp.size(); i++){
            InteractionHistory<A> ih = this.tmp.get(i);
            double error = errors.getDouble(i);
            ih.computeError(error);
            this.history.add(ih);
        }
        /*int i = 0 ;
        System.out.println("printing");
        for(InteractionHistory it : this.history){
            System.out.println(it.getErrorValue());
            if(i> 10)
                break ;
            i++;
        }*/
        this.tmp= new ArrayList<>();
    }

    @Override
    public Interaction<A> chooseInteraction() {
        if(this.interactions.size() > 0) {
            InteractionHistory<A> ih = this.history.pollLast();
            Interaction<A> i = ih.getInteraction();
            this.tmp.add(ih);
            //System.out.println("choose "+ih.getErrorValue());
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
        return this.history.size();
    }

    public void print(){
        for(InteractionHistory ih : this.history){
            System.out.println(ih.getErrorValue()+"  "+ih.getSumValues());
        }
    }
}
