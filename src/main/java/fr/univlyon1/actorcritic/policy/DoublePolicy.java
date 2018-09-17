package fr.univlyon1.actorcritic.policy;

import fr.univlyon1.learning.Informations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class DoublePolicy<A> implements Policy<A> {

    private Policy continuous;
    private Policy discrete ;

    public DoublePolicy(Policy continuous, Policy discrete){
        this.continuous = continuous;
        this.discrete = discrete;
    }

    @Override
    public Object getAction(INDArray results,Informations information) {
        INDArray cont = (INDArray)this.continuous.getAction(results,information);
        INDArray dis = (INDArray)this.discrete.getAction(results,information);
        INDArray res = Nd4j.zeros(cont.size(1));
        res.put(new INDArrayIndex[]{NDArrayIndex.interval(0,2)},cont.get(NDArrayIndex.interval(0,2)));
        if(dis.size(1) > 2)
            res.put(new INDArrayIndex[]{NDArrayIndex.point(2)},dis.get(NDArrayIndex.point(2)));
        return res;
    }
}
