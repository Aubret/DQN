package main.java.fr.univlyon1.environment;

import java.util.ArrayList;

public abstract class ActionSpace<A> {
    protected ArrayList<A> actions ;
    protected int seed ;

    public ActionSpace(){
        this.actions = new ArrayList<A>();
    }

    public abstract void addAction(A action);
    public abstract A mapNumberToAction(Object number);
    public abstract int mapActionToNumber(A action);

    public int getSize() {
        return actions.size();
    }

    public void setSeed(int i) {
        this.seed = i ;
    }

    public String toString(){
        String str = "";
        for(A action : actions){
            str+=action+"; ";
        }
        return str ;
    }

}

