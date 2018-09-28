package fr.univlyon1.environment.interactions;

import fr.univlyon1.configurations.PojoSpecificObservation;
import fr.univlyon1.environment.space.SpecificObservation;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MiniObs implements SpecificObservation{

    private Double time ;
    private long id ;
    private boolean alreadySent;
    private INDArray labels ;

    public MiniObs(SpecificObservation spo){
        this.time = spo.getOrderedNumber() ;
        this.id = spo.getId() ;
        this.alreadySent = spo.hasAlreadySent() ;
        this.labels = spo.getLabels() ;
    }

    public MiniObs(PojoSpecificObservation pspo){
        this.time = pspo.getTime();
        this.id = pspo.getId() ;
        this.alreadySent = pspo.isAlreadySent();
        this.labels = Nd4j.create(pspo.getLabels()) ;
    }

    public MiniObs(Interaction i){
        this.time = i.getTime() ;
    }

    @Override
    public INDArray getData() {
        return null;
    }

    @Override
    public void computeData() {

    }

    @Override
    public long getId() {
        return 0;
    }

    @Override
    public boolean hasAlreadySent() {
        return false;
    }

    @Override
    public Double getOrderedNumber() {
        return this.time;
    }

    @Override
    public INDArray getLabels() {
        return null;
    }

    @Override
    public int compareTo(@NotNull Object o) {
        if(o instanceof SpecificObservation){
            SpecificObservation other = (SpecificObservation)o ;
            if(this.getOrderedNumber() < other.getOrderedNumber() ){
                return -1 ;
            }else if(this.getOrderedNumber().equals(other.getOrderedNumber())){
                return 0 ;
            }else{
                return 1 ;
            }

        }else{
            System.out.println("error comparaison specificobs");
        }
        return 0 ;
    }
}
