# `tenants` endpoints

Routes registered under `/api/v1/tenants`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/tenants` |
| POST | `/tenants` |
| GET | `/tenants/all` |
| GET | `/tenants/kv/{key}` |
| PUT | `/tenants/kv/{key}` |
| GET | `/tenants/search` |
| DELETE | `/tenants/{tenant_id}` |
| GET | `/tenants/{tenant_id}` |
| PUT | `/tenants/{tenant_id}` |
| GET | `/tenants/{tenant_id}/api-keys` |
| POST | `/tenants/{tenant_id}/api-keys` |
| DELETE | `/tenants/{tenant_id}/api-keys/{key_id}` |
| GET | `/tenants/{tenant_id}/api-principal-config` |
| PUT | `/tenants/{tenant_id}/api-principal-config` |
| GET | `/tenants/{tenant_id}/members` |
| POST | `/tenants/{tenant_id}/members` |
| PUT | `/tenants/{tenant_id}/members/{user_id}` |
| DELETE | `/tenants/{tenant_id}/members/{user_id}` |
| POST | `/tenants/{tenant_id}/leave` |
| GET | `/tenants/{tenant_id}/invitations` |
| POST | `/tenants/{tenant_id}/invitations` |
| DELETE | `/tenants/{tenant_id}/invitations/{inv_id}` |
| POST | `/tenants/{tenant_id}/invite-links` |
