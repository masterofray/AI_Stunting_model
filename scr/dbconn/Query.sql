CREATE TABLE "Second_RawData" AS
SELECT
	B."District_ID",
	A."stunting",
	A."gender",
	A."malnutrition_level",
	A."exclusive_milky",
	A."smoke_habit",
	A."purewater_access",
	A."healthy_toilet",
	A."difficult_acess"
FROM 
	"rawdata" AS A
LEFT JOIN
	"DistrictID" AS B
ON
	A.district_name = B.district_name
ORDER BY
	"District_ID" ASC;

SELECT * FROM "rawdata";rawdatarawdata

SELECT DISTINCT "District_ID", "district_name" FROM "DistrictID" ORDER BY "District_ID";

CREATE TABLE "Label_District" AS SELECT DISTINCT "District_ID", "district_name" FROM "DistrictID" ORDER BY "District_ID";