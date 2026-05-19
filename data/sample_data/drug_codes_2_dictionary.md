Actual Medicinal Product Pack (AMPP) Level

- __record_tag_AMPP: Identifies the XML element source tag for the AMPP record.
- APPID: Unique SNOMED CT identifier for the Actual Medicinal Product Pack.
- APID: Unique identifier linking the pack to its parent Actual Medicinal Product.
- VPPID: Unique identifier linking to the parent Virtual Medicinal Product Pack.
- NM_AMPP: Full descriptive name of the Actual Medicinal Product Pack.
- SUBP: Information describing the sub-pack composition, such as "2 x 14 tablets".
- LEGAL_CATCD: Code indicating the legal prescribing category, such as General Sales List or Prescription Only Medicine.
- COMBPACKCD: Code denoting if the pack is a combination product or a component of one.
- DISCCD: Flag indicating if the supplier has discontinued the pack type.
- DISCDT: The date the discontinued status last changed.
- INVALID_AMPP: Flag denoting if the AMPP dictionary entry is invalid.

Actual Medicinal Product (AMP) Level

- __record_tag_AMP: Identifies the XML element source tag for the AMP record.
- VPID: Unique identifier linking the AMP to its parent Virtual Medicinal Product.
- NM_AMP: Full name identifying the Actual Medicinal Product.
- ABBREVNM: Abbreviated version of the AMP name, up to 60 characters.
- DESC: Unique description combining the name, order number, size, colour, and supplier.
- SUPPCD: SNOMED CT identifier for the product's supplier.
- LIC_AUTHCD: Code identifying the current licensing authority, such as MHRA or EMA.
- LIC_AUTHCHANGECD: Code detailing the reason for a licensing authority change.
- LIC_AUTHCHANGEDT: Date the licensing authority changed.
- LIC_AUTH_PREVCD: Code for the previous licensing authority.
- AVAIL_RESTRICTCD: Code identifying any restrictions on the availability of the AMP.
- COMBPRODCD: Flag indicating if the AMP is a combination product.
- EMA: Flag indicating if the product is under additional monitoring by the European Medicines Agency.
- PARALLEL_IMPORT: Flag indicating if the AMP has been imported from within the EU.
- FLAVOURCD: Code indicating the product flavour if it differentiates clinically equivalent AMPs.
- NMDT: Date the name became the preferred name for the AMP.
- NM_PREV: Previous name of the Actual Medicinal Product.
- INVALID_AMP: Flag denoting if the AMP dictionary entry is invalid.

Virtual Medicinal Product Pack (VMPP) Level

- __record_tag: Identifies the XML element source tag for the VMPP record.
- NM: Full descriptive name of the Virtual Medicinal Product Pack.
- QTYVAL: Numerical quantity of the VMP contained in the pack.
- QTY_UOMCD: SNOMED CT identifier for the quantity's unit of measure.
- INVALID: Flag denoting if the VMPP dictionary entry is invalid.

Virtual Medicinal Product (VMP) Level

- __record_tag_VMP: Identifies the XML element source tag for the VMP record.
- VTMID: Unique identifier linking the VMP to its parent Virtual Therapeutic Moiety.
- NM_VMP: Full name identifying the Virtual Medicinal Product.
- ABBREVNM_VMP: Abbreviated version of the VMP name.
- PRES_STATCD: Code indicating the prescribing status of the generic product.
- DF_INDCD: Dose form indicator classifying the product as discrete, continuous, or not applicable.
- UDFS: Numerical unit dose form size.
- UDFS_UOMCD: SNOMED CT identifier for the unit of measure relating to the dose size.
- UNIT_DOSE_UOMCD: SNOMED CT identifier representing the physical dose unit, such as tablet or vial.
- BASISCD: Code indicating the basis or source of the preferred name, like rINN or BAN.
- BASIS_PREVCD: Code indicating the basis or source of the previous name.
- COMBPRODCD_VMP: Flag indicating if the VMP is a combination product.
- NON_AVAILCD: Flag indicating if there are currently no available Actual Medicinal Products for this VMP.
- NON_AVAILDT: Date the non-availability status changed.
- CFC_F: Flag indicating if the formulation is CFC-free.
- GLU_F: Flag indicating if the formulation is gluten-free.
- PRES_F: Flag indicating if the formulation is preservative-free.
- SUG_F: Flag indicating if the formulation is sugar-free.
- NMCHANGECD: Code representing the reason for a VMP name change.
- NMDT_VMP: Date the name became the preferred name for the VMP.
- NMPREV: Previous name of the Virtual Medicinal Product.
- VPIDDT: Date the Virtual Medicinal Product identifier became applicable.
- VPIDPREV: Previously allocated identifier for the Virtual Medicinal Product.
- INVALID_VMP: Flag denoting if the VMP dictionary entry is invalid.

Virtual Therapeutic Moiety (VTM) Level

- __record_tag_VTM: Identifies the XML element source tag for the VTM record.
- NM_VTM: Full name identifying the abstract Virtual Therapeutic Moiety substance.
- ABBREVNM_VTM: Abbreviated version of the VTM name.
- VTMIDDT: Date the Virtual Therapeutic Moiety identifier became applicable.
- VTMIDPREV: Previously allocated identifier for the Virtual Therapeutic Moiety.
- INVALID_VTM: Flag denoting if the VTM dictionary entry is invalid.

Aggregated Features

- AGGREGATED_INGREDIENTS: A human-readable string combining all active ingredient names, strengths, and units associated with the VMP, separated by " | ".
- AGGREGATED_ROUTES: A human-readable string combining all decoded routes of administration associated with the VMP.
- DOSE_FORM: The decoded, textual name of the physical administration form.
- SUPPLIER_NAME: The decoded, textual name of the supplier associated with the AMP.