package fr.univlyon1.memory;

import fr.univlyon1.environment.Interaction;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;


public class SequentialExperienceReplay<A> extends ExperienceReplay<A>{
    protected ArrayList<Interaction<A>> interactions ;
    protected ArrayList<Interaction<A>> tmp ;
    protected Integer cursor=0;
    protected int sequenceSize ;
    protected int backpropSize ;

    protected int forwardNumber ;
    protected int backpropNumber;
    protected double startTime ;

    public SequentialExperienceReplay(int maxSize, ArrayList<String> file,int sequenceSize, int backpropSize){
        super(maxSize,file);
        this.resetMemory();
        this.sequenceSize = sequenceSize ;
        this.backpropSize = backpropSize ;
    }

    @Override
    public void addInteraction(Interaction<A> interaction) {
        if(this.interactions.size() == this.maxSize-1)
            this.interactions.remove(this.interactions.get(0));
        this.interactions.add(interaction);
    }

    public boolean initChoose(){ // Toujours appeler avant les chooseInteraction
        if(this.interactions.size() == 0)
            return false ;
        if(cursor == this.interactions.size()-1)
            cursor = 0 ;
        // On vérifie qu ele curseur actuel suffit à proposer une séquence complète
        Interaction<A> start = this.interactions.get(cursor);
        Double dt = this.interactions.get(this.interactions.size()-1).getTime() - start.getTime() ;
        if(dt < this.sequenceSize){
            if(cursor == 0)
                return false ;
            else
                cursor = 0;
        }
        this.startTime = this.interactions.get(cursor).getTime() ;
        this.backpropNumber = 0 ;
        this.forwardNumber = 0 ;
        this.tmp = new ArrayList<>();
        return true ;
    }

    @Override
    public Interaction<A> chooseInteraction() {
        Interaction<A> choose = this.interactions.get(cursor);
        Double dt = choose.getTime() - startTime;
        if(dt > sequenceSize)
            return null;
        tmp.add(choose);
        this.forwardNumber++ ;
        this.cursor++ ;
        this.tmp.add(choose);
        return choose;
    }

    @Override
    public void resetMemory() {
        this.interactions = new ArrayList<>();
        this.tmp = new ArrayList<>();
    }

    @Override
    public int getSize() {
        return this.interactions.size();
    }

    @Override
    public void setError(INDArray errors) {

    }


    public int getForwardNumber(){
        return this.forwardNumber ;
    }

    public int getBackpropNumber(){
        Double endTime = this.tmp.get(this.tmp.size()-1).getTime() ;
        for(int i = this.tmp.size()-1 ; i >= 0 ; i-- ){
            if(endTime - this.tmp.get(i).getTime() > this.backpropSize){
                return this.backpropNumber ;
            }else
                this.backpropNumber++ ;

        }
        return this.backpropNumber;
    }

}
