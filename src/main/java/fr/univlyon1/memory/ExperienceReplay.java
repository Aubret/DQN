package fr.univlyon1.memory;

import fr.univlyon1.environment.Interaction;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public abstract class ExperienceReplay<A> {

    protected int maxSize ;
    public ExperienceReplay(int maxSize){
        this.maxSize = maxSize ;
    }

    public abstract void addInteraction(Interaction<A> interaction);
    public abstract Interaction<A> chooseInteraction();
    public abstract void resetMemory();
    public abstract int getSize();
    public abstract void setError(INDArray errors);

    public int getMaxSize() {
        return maxSize;
    }

    public void setMaxSize(int maxSize) {
        this.maxSize = maxSize;
    }
}
