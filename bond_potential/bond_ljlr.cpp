/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include "bond_ljlr.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondLjlr::BondLjlr(LAMMPS *lmp) : Bond(lmp) {}

/* ---------------------------------------------------------------------- */

BondLjlr::~BondLjlr()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(epsilon);
    memory->destroy(sigma);
	memory->destroy(rcut);
  }
}

/* ---------------------------------------------------------------------- */

void BondLjlr::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq;
  double r2inv,r3inv,r6inv,forcelj;
  double lj1, lj2, lj3, lj4, ratio, offset;

  ebond = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;

	r2inv = 1.0/rsq;
	r6inv = pow(r2inv,0.4);
	r3inv = r6inv;
	lj1 = 19.2 * epsilon[type] * pow(sigma[type],1.6);
	lj2 = 9.6 * epsilon[type] * pow(sigma[type],0.8);
	lj3 = 12.0 * epsilon[type] * pow(sigma[type],1.6);
	lj4 = 12.0 * epsilon[type] * pow(sigma[type],0.8);
	ratio = sigma[type] / rcut[type]; 
	offset = 12.0 * epsilon[type] * (pow(ratio,1.6) - pow(ratio,0.8));
	forcelj = r6inv*(lj1*r3inv - lj2);

    // force & energy

    if (rsq > 0.0) fbond = forcelj*r2inv;
    else fbond = 0.0;

    if (eflag) ebond = r6inv*(lj3*r3inv-lj4) - offset;

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond;
      f[i1][1] += dely*fbond;
      f[i1][2] += delz*fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond;
      f[i2][1] -= dely*fbond;
      f[i2][2] -= delz*fbond;
    }

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondLjlr::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(epsilon,n+1,"bond:epsilon");
  memory->create(sigma,n+1,"bond:sigma");
  memory->create(rcut,n+1,"bond:rcut");
  
  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondLjlr::coeff(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nbondtypes,ilo,ihi);

  double epsilon_one = force->numeric(FLERR,arg[1]);
  double sigma_one = force->numeric(FLERR,arg[2]);
  double rcut_one = force->numeric(FLERR,arg[3]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    epsilon[i] = epsilon_one;
    sigma[i] = sigma_one;
	rcut[i] = rcut_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondLjlr::equilibrium_distance(int i)
{
  return 2.378*sigma[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondLjlr::write_restart(FILE *fp)
{
  fwrite(&epsilon[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&sigma[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&rcut[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondLjlr::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&epsilon[1],sizeof(double),atom->nbondtypes,fp);
    fread(&sigma[1],sizeof(double),atom->nbondtypes,fp);
	fread(&rcut[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&epsilon[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&sigma[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&rcut[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondLjlr::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g\n",i,epsilon[i],sigma[i],rcut[i]);
}

/* ---------------------------------------------------------------------- */

double BondLjlr::single(int type, double rsq, int i, int j,
                        double &fforce)
{

    double r2inv = 1.0/rsq;
	double r6inv = pow(r2inv,0.4);
	double r3inv = r6inv;
	double lj1 = 19.2 * epsilon[type] * pow(sigma[type],1.6);
	double lj2 = 9.6 * epsilon[type] * pow(sigma[type],0.8);
	double lj3 = 12.0 * epsilon[type] * pow(sigma[type],1.6);
	double lj4 = 12.0 * epsilon[type] * pow(sigma[type],0.8);
	double ratio = sigma[type] / rcut[type]; 
	double offset = 12.0 * epsilon[type] * (pow(ratio,1.6) - pow(ratio,0.8));
	double forcelj = r6inv*(lj1*r3inv - lj2);
    fforce = 0.0;
	if (rsq > 0.0) fforce = forcelj*r2inv;
	
	double eng= r6inv*(lj3*r3inv-lj4)-offset; 
	
    return eng; 
  
}
