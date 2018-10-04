package fr.univlyon1.selfsupervised.dataTransfer;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.selfsupervised.dataConstructors.LstmDataConstructors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DataNothing<A> extends DataList<A> {
    public DataNothing(SpecificObservation observation, Interaction<A> predictions, Double extratime, LstmDataConstructors<A> ldc) {
        super(observation, predictions, extratime, ldc);
    }

    public INDArray constructAddings(){
        return Nd4j.create(new double[]{this.extratime});
    }

}
