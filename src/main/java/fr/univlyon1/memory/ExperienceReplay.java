package fr.univlyon1.memory;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;

public abstract class ExperienceReplay<A> {
    protected ArrayList<String> file ;
    protected int maxSize ;


    public ExperienceReplay(int maxSize,ArrayList<String> file){
        this.file = file ;
        this.maxSize = maxSize ;
    }

    public abstract void addInteraction(Interaction<A> interaction);
    public abstract Interaction<A> chooseInteraction();
    public abstract void resetMemory();
    public abstract int getSize();
    public abstract void setError(INDArray errors);

    public boolean initChoose(){return true ;}

    public int getMaxSize() {
        return maxSize;
    }

    public void setMaxSize(int maxSize) {
        this.maxSize = maxSize;
    }

    public void load(ActionSpace<A> as){
        /*if(file != null){ // Need to adjust gamma interaction
            for(String f : this.file) {
                try {
                    JAXBContext context = JAXBContext.newInstance(ListPojo.class);
                    Unmarshaller unmarshaller = context.createUnmarshaller();
                    ListPojo<A> lp = (ListPojo) unmarshaller.unmarshal(new File(f));
                    for (PojoInteraction<A> pi : lp.getPojos()) {
                        Interaction<A> i = new Interaction<A>(as.mapNumberToAction(Nd4j.create(pi.getAction())), Nd4j.create(pi.getObservation()));
                        i.setReward(pi.getReward());
                        i.setSecondObservation(Nd4j.create(pi.getSecondObservation()));
                        this.addInteraction(i);
                        if (this.getSize() == this.maxSize - 1)
                            return;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            this.setError(null);
        }*/
    }
}
