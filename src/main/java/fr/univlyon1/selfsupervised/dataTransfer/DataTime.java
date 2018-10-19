package fr.univlyon1.selfsupervised.dataTransfer;

import fr.univlyon1.environment.interactions.Interaction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DataTime<A> implements DataTarget<A> {
    protected Interaction<A> predictions ;

    public DataTime(Interaction<A> predictions){
        this.predictions = predictions ;
    }

    @Override
    public INDArray getLabels() {
        return Nd4j.create(new double[]{predictions.getDt()});
    }

    @Override
    public INDArray constructAddings() {
        return Nd4j.create(new double[]{});
    }
}
