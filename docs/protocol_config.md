# Protocol Config (v1.1)

See [config/protocol.yaml](../config/protocol.yaml) for the machine-readable source.

**Changelog v1.1:** Drop TWN and PPI; CPI from FRED/OECD monthly index; trade from CEPII BACI bulk; policy_rate forward-fill ≤2Q.

```yaml
protocol_version: "1.1"
countries: 23  # G20 + NLD, SGP, CHE, VNM (no TWN)
features: [cpi_yoy, gdp_yoy, neer_chg, policy_rate, energy_idx, covid]
cpi_source: FRED monthly + OECD fallback
trade_source: CEPII BACI bulk CSV
```
