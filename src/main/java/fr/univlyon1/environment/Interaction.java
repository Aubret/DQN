package fr.univlyon1.environment;

import fr.univlyon1.environment.space.ContinuousAction;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

@Setter
@Getter
public class Interaction <A>{
    protected static int count = 0 ;
    protected Double time ;
    protected INDArray observation ;
    protected INDArray secondObservation ;

    protected INDArray state ;
    protected INDArray secondState ;
    protected Object memoryBefore ;
    protected Object memoryAfter ;


    protected A action ;
    protected A secondAction ;
    protected double reward ;
    protected int id ;

    public Interaction(A action, INDArray observation){
        this.action = action ;
        this.observation = observation ;
        this.id = count ;
        count++;
        this.state = null ;
    }

    public Interaction<A> clone(){
        //A action = (A)((ContinuousAction)this.getAction()).copy();
        Interaction<A> i = new Interaction<A>(this.getAction(),this.getObservation());
        i.setSecondObservation(this.getSecondObservation());
        i.setReward(this.getReward());
        i.setId(this.getId());
        return i ;
    }

    public double computeReward() {
        return this.reward;
    }
}
