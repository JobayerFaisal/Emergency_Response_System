// frontend/types/dispatch.ts

export interface DispatchOrder {
  team_id: string;
  team_name: string;
  target_lat: number;
  target_lon: number;
  requester_name: string;
  requester_phone: string;
  details: string;
}
