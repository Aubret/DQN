package fr.univlyon1.environment;

public class DiscreteActionSpace<A> extends ActionSpace<A> {
    public DiscreteActionSpace(){
        super();
    }

    public void addAction(A action){
        this.actions.add(action);
    }

    public A mapNumberToAction(Object number){
        return this.actions.get((int)number);
    }

    @Override
    public Object mapActionToNumber(A action) {
        for(int i = 0 ; i < this.actions.size(); i++){
            if(this.actions.get(i) == action)
                return i ;
        }
        return -1 ;
    }
}
