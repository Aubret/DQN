package fr.univlyon1.environment;

import fr.univlyon1.environment.space.ContinuousAction;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

@Setter
@Getter
public class Interaction <A>{
    private static int count = 0 ;
    private INDArray observation ;
    private INDArray secondObservation ;

    private INDArray state =null ;
    private INDArray secondState=null ;

    private A action ;
    private A secondAction ;
    private double reward ;
    private int id ;

    public Interaction(A action, INDArray observation){
        this.action = action ;
        this.observation = observation ;
        this.id = count ;
        count++;
    }

    public Interaction(A action, INDArray observation, INDArray state){
        this(action,observation);
        this.setState(state);
    }

    public Interaction<A> clone(){
        //A action = (A)((ContinuousAction)this.getAction()).copy();
        Interaction<A> i = new Interaction<A>(this.getAction(),this.getObservation());
        i.setSecondObservation(this.getSecondObservation());
        i.setReward(this.getReward());
        i.setId(this.getId());
        return i ;
    }
}
