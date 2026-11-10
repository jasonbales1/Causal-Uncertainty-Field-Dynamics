# Third-Party Observational Datasets (`cufg/data/`)

All observational data files in `cufg/data/` are publicly released astronomical catalogs and FITS profiles produced by external research collaborations. They are cached here solely for offline reproducibility of the benchmark scripts (`cufg/benchmarks/galaxy_rotation.py` and `cufg/benchmarks/cosmology.py`). Full credit belongs to their original authors:

## 1. SPARC (Spitzer Photometry and Accurate Rotation Curves)
- **Files:**
  - `sparc_table1_galaxies.csv` (`175` disk galaxies from VizieR `J/AJ/152/157/table1`)
  - `sparc_table2_massmodels.csv` (`3,391` radial rotation-curve and mass-model rows from VizieR `J/AJ/152/157/table2`)
- **Authors & Credit:** Federico Lelli, Stacy S. McGaugh, and James M. Schombert
- **Required Citations:**
  - Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016), *"SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves,"* The Astronomical Journal, **152**, 157 (`doi:10.3847/0004-6256/152/6/157`, `arXiv:1606.09251`).
  - McGaugh, S. S., Lelli, F., & Schombert, J. M. (2016), *"Radial Acceleration Relation in Rotationally Supported Galaxies,"* Physical Review Letters, **117**, 201101 (`doi:10.1103/PhysRevLett.117.201101`, `arXiv:1609.05917`).
- **Original Archive:** CDS VizieR `J/AJ/152/157` (`https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/152/157`) and `http://astroweb.cwru.edu/SPARC/`.

## 2. X-COP (XMM-Newton Cluster Outskirts Project)
- **Files:**
  - `xcop_fits/*_fgas_profile.fits`, `xcop_fits/*_hydro_mass.fits`, `xcop_fits/*_mstar.fits` (`31` official FITS files across the 12 X-COP galaxy clusters: `A85`, `A644`, `A1644`, `A1795`, `A2029`, `A2142`, `A2255`, `A2319`, `A3158`, `A3266`, `RXC1825`, `ZW1215`)
  - `xcop_12clusters_radial_profiles.csv` (`600` radial profile rows extracted directly from `xcop_fits/*.fits`)
  - `xcop_ettori2019_table1.csv` (Table 1 of Ettori et al. 2019, `arXiv:1805.00035` `m_over.tex`)
  - `xcop_eckert2019_table2.csv` (Table 2 of Eckert et al. 2019, `arXiv:1805.00034` `table2.tex`)
- **Authors & Credit:** The X-COP Collaboration (Dominique Eckert, Stefano Ettori, Vittorio Ghirardini, et al.)
- **Required Citations:**
  - Ettori, S., Ghirardini, V., Eckert, D., et al. (2019), *"Hydrostatic mass profiles in X-COP galaxy clusters,"* Astronomy & Astrophysics, **621**, A39 (`doi:10.1051/0004-6361/201833323`, `arXiv:1805.00035`).
  - Eckert, D., Ghirardini, V., Ettori, S., et al. (2019), *"Non-thermal pressure support in X-COP galaxy clusters,"* Astronomy & Astrophysics, **621**, A40 (`doi:10.1051/0004-6361/201833324`, `arXiv:1805.00034`).
  - Ghirardini, V., Eckert, D., Ettori, S., et al. (2019), *"Universal thermodynamic properties of the intracluster medium over two decades in radius in the X-COP sample,"* Astronomy & Astrophysics, **621**, A41 (`doi:10.1051/0004-6361/201833325`).
- **Original Archive:** University of Geneva X-COP Public Data Release (`https://dominiqueeckert.wixsite.com/xcop/data-products`).

## 3. Planck 2018 (PR3) Cosmic Microwave Background Power Spectra
- **Files:**
  - `planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt` (`83` binned $TT$ bandpowers across $30 \le \ell \le 2500$)
  - `planck2018/COM_PowerSpect_CMB-TE-binned_R3.02.txt` (`66` binned $TE$ bandpowers across $30 \le \ell \le 1996$)
  - `planck2018/COM_PowerSpect_CMB-EE-binned_R3.02.txt` (`66` binned $EE$ bandpowers across $30 \le \ell \le 1996$)
  - `planck2018/COM_PowerSpect_CMB-TT-full_R3.01.txt` (`2,507` unbinned $TT$ multipoles across $2 \le \ell \le 2508$)
  - `planck2018/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt` (`2,507` multipoles of the official Planck 2018 PR3 reference $TT, TE, EE, BB, PP$ spectra)
- **Authors & Credit:** The Planck Collaboration (ESA / Planck Science Team)
- **Required Citations:**
  - Planck Collaboration (Aghanim, N., et al.) (2020), *"Planck 2018 results. V. CMB power spectra and likelihoods,"* Astronomy & Astrophysics, **641**, A5 (`doi:10.1051/0004-6361/201936386`, `arXiv:1907.12875`).
  - Planck Collaboration (Aghanim, N., et al.) (2020), *"Planck 2018 results. VI. Cosmological parameters,"* Astronomy & Astrophysics, **641**, A6 (`doi:10.1051/0004-6361/201833910`, `arXiv:1807.06209`).
- **Original Archive:** ESA Planck Legacy Archive (`http://pla.esac.esa.int`) and NASA/IPAC Infrared Science Archive (IRSA) Planck Release 3 Ancillary Data (`https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/`).
