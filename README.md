# ran-slice-scheduler-simulator
An event-driven simulator to simulate RAN slice schedulers.

## Simulator code

The simulator consists in a set of cells and UEs. Each cell has a number of slices configured and serves a dynamic set of UEs. At each scheduling opportunity, the cell must distribute its resources over the different slices (by using the inter-slice scheduler) and each slice serves its UEs with the given resources.

### Simulator.py

Includes the main simulator body.

### Cell.py

Includes the operations done at the Cell. The most imortant is calling the scheduling algorithms at each TTI.

### UE.py

Includes the operations done at each UE. Some examples are: movement, cell attachment, traffic activity, HARQ processes.

### IntraScheduler.py

Defines scheduling algorithms for the intra-slice schedulers. That is the schedulers that schedule UEs by using the resources given by the (inter-)slice scheduler

### Scheduler.py

Defines scheduling algorithms for the (inter-)slice schedulers. They are implemented in classes, and the class to be used should be specified in the configurations


### Util.py Util_sinr.py

Additional utilitarian functions
