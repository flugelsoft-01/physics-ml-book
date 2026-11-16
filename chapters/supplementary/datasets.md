# Sensor Telemetry & Open Physical Benchmark Datasets

While every capstone in this volume features a self-contained synthetic physics generator to ensure zero broken external dependencies, transitioning to real-world industrial validation requires accessing curated open physics and telemetry repositories.

Below is the verified master catalog of official open-access physical benchmark datasets across energy, aerospace, manufacturing, earth sciences, and healthcare.

---

## 1. Energy, Utilities & Thermal Systems

| Dataset Name | Curating Agency / Institution | Domain & Physical Content | Data Modality & Format | Access Protocol / URL |
| :--- | :--- | :--- | :--- | :--- |
| **NREL WIND Toolkit** | National Renewable Energy Laboratory | High-resolution 2km atmospheric boundary layer wind velocity, temperature, and pressure fields across the continental US. | HDF5, NetCDF, AWS S3 Open Data | `s3://nrel-pds-wtk/` |
| **NASA Ames Battery Aging Repository** | NASA Prognostics Center of Excellence | Li-ion 18650 cell cycling telemetry across accelerated degradation protocols with electrochemical impedance spectroscopy (EIS). | MATLAB `.mat`, CSV | `ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/` |
| **NREL NSRDB (Solar Radiation)** | National Renewable Energy Laboratory | 4km gridded Direct Normal Irradiance (DNI), Global Horizontal Irradiance (GHI), and optical aerosol depths from geostationary satellites. | REST API, HDF5 | `developer.nrel.gov/docs/solar/nsrdb/` |
| **PJM Synchrophasor Telemetry** | PJM Interconnection & DOE | Phasor Measurement Unit (PMU) 30–60 Hz high-frequency electrical bus frequency, voltage angles, and transmission line active power. | IEEE C37.118 streaming, CSV | Open EI Grid Data / DOE |

---

## 2. Aerospace, Defense & Automotive

| Dataset Name | Curating Agency / Institution | Domain & Physical Content | Data Modality & Format | Access Protocol / URL |
| :--- | :--- | :--- | :--- | :--- |
| **NASA OpenVSP Airfoil & Wing Datasets** | NASA Langley Research Center | 3D subsonic and supersonic aerodynamic pressure coefficients, lift, wave drag, and surface mesh geometries. | XML, STL, OpenVSP `.vsp3` | `openvsp.org` |
| **Space-Track.org Satellite Catalog** | US Space Command (18th Space Defense) | Two-Line Element (TLE) and Conjunction Data Messages (CDM) for 45,000+ active satellites and orbital debris fragments. | JSON REST API, TLE format | `space-track.org` |
| **NASA Turbofan Engine Degradation (C-MAPSS)** | NASA Ames Research Center | Run-to-failure vibration, pressure, and thermal sensor streams from 21 turbofan engine telemetry channels across altitude cycles. | Tabular text, CSV | NASA Prognostics Repository |
| **nuScenes 4D Radar & LiDAR Dataset** | Motional | Full multimodal autonomous vehicle sensor suite with 32-beam LiDAR point clouds and 4D imaging radar velocity vectors. | JSON, binary `.pcd`, ROS bags | `nuscenes.org` |

---

## 3. Materials, Manufacturing & Semiconductors

| Dataset Name | Curating Agency / Institution | Domain & Physical Content | Data Modality & Format | Access Protocol / URL |
| :--- | :--- | :--- | :--- | :--- |
| **The Materials Project** | Lawrence Berkeley National Laboratory | DFT-calculated formation energies, bandgaps, crystal structures (CIF), and phonon density of states for 150,000+ inorganic compounds. | REST API (`mp-api`), JSON, CIF | `materialsproject.org` |
| **NIST Superconducting Materials Database** | National Institute of Standards & Tech | Evaluated critical temperatures ($T_c$), critical magnetic fields ($H_{c2}$), and chemical stoichiometries for superconductor alloys. | CSV, NIST Standard Reference | `data.nist.gov` |
| **WM-811K Wafer Map Dataset** | Taiwan Semiconductor Reliability Lab | 811,457 real fabrication semiconductor wafer maps with expert-labeled defect patterns (Center, Donut, Scratch, Edge-Ring). | Pickle, NumPy `.npy` | Academic open mirror |
| **NIST Additive Manufacturing Benchmark (AM-Bench)** | NIST Engineering Laboratory | High-speed infrared thermal video and in-situ X-ray radiographs of titanium laser powder bed fusion (L-PBF) melt-pool keyhole dynamics. | TIFF image stacks, NetCDF | `nist.gov/ambench` |

---

## 4. Earth Sciences, Ocean & Infrastructure

| Dataset Name | Curating Agency / Institution | Domain & Physical Content | Data Modality & Format | Access Protocol / URL |
| :--- | :--- | :--- | :--- | :--- |
| **USGS Northern California Seismic Network** | US Geological Survey (USGS) | Broadband 100 Hz seismograph waveforms with precise $P$-wave and $S$-wave arrival pick catalogs and hypocenter coordinates. | MiniSEED, QuakeML, FDSN Web API | `service.iris.edu/fdsnws/dataselect/1/` |
| **Copernicus CMEMS Ocean Altimetry** | European Space Agency (ESA) & Mercator Ocean | Global gridded daily sea surface heights ($\eta$), geostrophic surface currents ($u, v$), and Stokes drift wave fields at $0.08^\circ$ resolution. | NetCDF-4, OPeNDAP | `marine.copernicus.eu` |
| **Open-Elevation Global 30m DEM** | NASA SRTM & USGS | Global Digital Elevation Model (DEM) topographic grids for catchment runoff, slope gradients, and hydrodynamic inundation modeling. | GeoTIFF, REST API | `open-elevation.com` |
| **EPA Air Quality System (AQS)** | US Environmental Protection Agency | Hourly ground-level greenhouse gas concentrations ($\text{CO}_2, \text{CH}_4, \text{NO}_2, \text{SO}_2$) across industrial fence-line monitor stations. | CSV, EPA AQS REST API | `aqs.epa.gov/aqsweb/documents/data_api.html` |

---

## 5. Biomedical, Healthcare & Complex Systems

| Dataset Name | Curating Agency / Institution | Domain & Physical Content | Data Modality & Format | Access Protocol / URL |
| :--- | :--- | :--- | :--- | :--- |
| **fastMRI Dataset** | NYU Langone Health & Meta AI | Largest public raw raw $k$-space MRI dataset containing 1,500+ fully-sampled clinical knee and brain acquisitions from 1.5T and 3.0T scanners. | HDF5 (complex $k$-space arrays) | `fastmri.med.nyu.edu` |
| **Aneurisk Cerebral Vascular Morphology** | Politecnico di Milano & Emory University | 3D surface meshes and pulsatile inlet flow profiles for 100+ patient-specific cerebral aneurysm geometries reconstructed from 3D rotational angiography. | VTK (`.vmtk`), STL, CSV | `ecm2.mathcs.emory.edu/aneurisk/` |
| **LOBSTER High-Frequency Limit Order Book** | Humboldt University Berlin | Nanosecond-timestamped limit order submissions, cancellations, and executions across all price levels for NASDAQ equities. | CSV, binary tick data | `lobsterdata.com` |
| **Caltrans PeMS Highway Sensor Network** | California Department of Transportation | 30-second loop detector vehicular flow ($q$), occupancy, and speed ($v$) across 39,000 spatial highway lane detectors. | Tabular text, CSV | `pems.dot.ca.gov` |
