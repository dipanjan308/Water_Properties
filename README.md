## number_h_bonds_ding_criterion_nvt.py: 
Calculates average number of hydrogen bonds per oxygen atom for fixed box size in NVT ensamble. It works through the following steps. 
For each hydrogen, the first and second nearest oxygen are determined. Nearest oxygen has O-H bond while the second nearest oxygen could 
be connected via hydrogen bond (HB). To define a HB we use three criterion: 
(1) The angle between acceptor oxygen, donor oxygen and the hydrogen atom is less than 30 degrees.
(2) The distance between the acceptor oxygen and the donor oxygen is less than 3.4 Angstorm.
(3) The distance between the hydrogen and the acceptor oxygen is less than 2.5 Angstorm.

## number_h_bonds_ding_criterion_npt.py: 
Calculates average number of hydrogen bonds per oxygen atom for varying box size in NPT ensamble.

## static_sf_pw.py:
Calculates the static structure factor between oxygen atoms of water.

## q4.py:
Calculates the probability distribution of the tetrahedral order parameter q4. It determines four nearest neighbor oxygen atoms of a oxygen atom and calculates q4.

## oho_angle_dist.py:
Calculates the probability distribution of O-H..O angle for all hydrogen atoms present in the system. First it finds two nearest neighbor oxygen atoms of each hydrogen atom. Then it calculates the angle made by oxygen1-hydrogen-oxygen2. 
