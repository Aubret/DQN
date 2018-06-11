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
    protected int num;


    public StochasticPrioritizedExperienceReplay(int maxSize,long seed,ArrayList<String> file) {
        super(maxSize,file);
        this.resetMemory();
        this.random = new Random(seed);
        this.num = 0 ;
    }
    @Override
    public void addInteraction(Interaction<A> interaction) {
        if(this.num != 5) {
            this.num++ ;
            return;
        }
        this.num=0;
        if(this.history.size() == this.maxSize) {
            InteractionHistory ih = history.getFirst();
            this.interactions.remove(ih.getInteraction());
        }
        InteractionHistory<A> newIh = new InteractionHistory<A>(interaction,this.random.nextDouble());
        toTake.add(newIh);
        //System.out.println("----");
        //this.history.insert(newIh);
        this.interactions.put(interaction, newIh);
    }

    // Ne pas combiner Not taken et le classique
    public void addInteractionNotTaken(Interaction<A> interaction) {
        InteractionHistory<A> newIh = new InteractionHistory<A>(interaction,0.1);
        this.history.insert(newIh);
        this.interactions.put(interaction, newIh);
    }

    public void removeInteraction(Interaction<A> remove){
        if(remove != null) {
            InteractionHistory ih = this.interactions.get(remove);
            if(ih != null) {
                InteractionHistory ih2=this.history.getInteractionUp(ih.getErrorValue(), ih.getId());
                if(ih2 == null){
                    System.out.println("null");
                }else if(ih2 != ih ){
                    System.out.println("differen !");
                }
                this.interactions.remove(remove);
            }
        }
    }

    public void repushLast(){
        InteractionHistory<A> ih = this.tmp.get(this.tmp.size()-1);
        this.history.insert(ih);
        this.tmp.remove(this.tmp.size()-1);
    }

    @Override
    public void setError(INDArray errors) {
        if(errors == null)
            return ;
        for(int i = 0;i< this.tmp.size(); i++){
            InteractionHistory<A> ih = this.tmp.get(i);
            if(this.interactions.containsKey(ih.getInteraction())) {
                double error = errors.getDouble(i);
                ih.computeError(error); // Important de le faire avant
                this.history.insert(ih);
            }
        }
        this.tmp = new ArrayList<>();
    }

    public boolean initChoose(){
        return true ;
    }

    @Override
    public Interaction<A> chooseInteraction() {
        if(this.random.nextBoolean() || this.history.size() == 0) {
            if (this.toTake.size() != 0) {
                InteractionHistory<A> ih = this.toTake.remove(toTake.size() - 1);
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

    public void print(){
        this.history.print();
    }
}
