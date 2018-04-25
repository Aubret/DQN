package fr.univlyon1.environment;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

@Setter
@Getter
public class Interaction <A> implements Cloneable{
    private static int count = 0 ;
    private INDArray observation ;
    private INDArray secondObservation ;

    private INDArray state ;
    private INDArray secondState ;
    private INDArray memory ;

    private A action ;
    private A secondAction ;
    private double reward ;
    private int id ;

    public Interaction(A action, INDArray observation){
        this.action = action ;
        this.observation = observation ;
        this.id = count ;
        count++;
        this.state = null ;
    }

    public Interaction(A action, INDArray observation,INDArray state){
        this(action, observation);
        this.state = state ;
    }

    public Interaction<A> clone(){
        Interaction<A> i = new Interaction<A>(this.getAction(),this.getObservation());
        i.setSecondObservation(this.getSecondObservation());
        i.setReward(this.getReward());
        i.setId(this.getId());
        return i ;
    }
}
