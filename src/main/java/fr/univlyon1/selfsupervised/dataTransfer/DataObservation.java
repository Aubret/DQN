package fr.univlyon1.selfsupervised.dataTransfer;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.selfsupervised.dataConstructors.DataConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DataObservation<A> implements DataTarget<A> {
    protected SpecificObservation observation ;
    protected Interaction<A> predictions ;
    protected Double extratime ;
    protected DataConstructor<A> ldc ;

    public DataObservation(SpecificObservation observation, Interaction<A> predictions, Double extratime, DataConstructor<A> ldc){
        this.observation = observation ;
        this.predictions = predictions ;
        this.extratime = extratime ;
        this.ldc = ldc ;
    }

    @Override
    public INDArray getLabels() {
        return observation.getLabels();
    }

    @Override
    public INDArray constructAddings() {
        //return Nd4j.concat(1,Nd4j.create(new double[]{this.extratime}),predictions.getObservation(),(INDArray)ldc.getActionSpace().mapActionToNumber(predictions.getAction()));
        return Nd4j.concat(1,Nd4j.create(new double[]{this.extratime}),predictions.getObservation());
    }
}
