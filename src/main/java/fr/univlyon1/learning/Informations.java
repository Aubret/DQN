package fr.univlyon1.learning;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

@Getter
@Setter
public class Informations {

    protected INDArray evaluatedActions ;
    protected INDArray evaluatedInputs;
    protected double score ;
    protected double dt ;
    protected boolean modified ;

    public Informations(){
        this.dt= 0.;
        this.modified= true ;
    }

}
