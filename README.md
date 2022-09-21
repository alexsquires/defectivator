# defectivator

`defectivator` is an API for generating defects in periodic crystalline materials. It is very much a work in progress, some general features include:

- all symmetry dealt with in the primitive cell and then rescaled to the defect cell for speed
- defects can be identified and the origin of the unit cell shifted such that they are centered for visulatisation purposes
- defect complex generation
- calculate sensible charge states for point defects using [data-mined oxidation states](https://pubs.acs.org/doi/pdf/10.1021/acs.jpclett.0c02072) 
