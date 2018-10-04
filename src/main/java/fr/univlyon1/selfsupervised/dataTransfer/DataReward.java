package fr.univlyon1.selfsupervised.dataTransfer;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.SpecificObservation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DataReward<A> implements DataTarget<A> {

    protected Interaction<A> predictions ;

    public DataReward(Interaction<A> predictions){
        this.predictions = predictions ;
    }

    public INDArray getLabels(){ return Nd4j.create(new double[]{this.predictions.getReward()}); }

    public INDArray constructAddings(){
        return Nd4j.create(new double[]{this.predictions.getDt()});
    }

}
