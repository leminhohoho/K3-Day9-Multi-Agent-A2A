from pydantic import BaseModel, Field
from typing import Optional


class OrderItem(BaseModel):
    item_id: int
    seller_id: str
    price: float
    freight_value: float
    shipping_limit_ts: Optional[str] = None


class OrderFinding(BaseModel):
    order_status: str
    purchase_ts: Optional[str] = None
    approved_ts: Optional[str] = None
    delivered_carrier_ts: Optional[str] = None
    delivered_customer_ts: Optional[str] = None
    estimated_delivery_ts: Optional[str] = None
    items: list[OrderItem] = []
    sellers: list[str] = []
    item_total_brl: float = 0.0
    freight_total_brl: float = 0.0


class PaymentRow(BaseModel):
    sequential: int
    type: str
    value: float


class PaymentFinding(BaseModel):
    payment_rows: list[PaymentRow] = []
    payment_total_brl: float = 0.0
    expected_total_brl: float = 0.0
    reconciled: bool = False
    discrepancy_brl: float = 0.0


class DeliveryFinding(BaseModel):
    delivered_late: bool = False
    late_days: float = 0.0
    carrier_received_late: bool = False
    responsible: str = "none"
    candidate_cause: str = ""


class RootCause(BaseModel):
    cause_code: str
    rank: int


class ResponsibleParty(BaseModel):
    party_type: str
    party_id: str


class Assessment(BaseModel):
    primary_issue: str
    case_status: str  # "action_required" | "no_action"
    confidence: float  # [0, 1]


class AffectedEntities(BaseModel):
    order_ids: list[str] = []
    item_ids: list[str] = []
    seller_ids: list[str] = []
    payment_ids: list[str] = []


class RootCauseAnalysis(BaseModel):
    ranked_causes: list[RootCause] = []
    responsible_parties: list[ResponsibleParty] = []


class FinancialResolution(BaseModel):
    currency: str = "BRL"
    item_total_brl: float = 0.0
    freight_total_brl: float = 0.0
    payment_total_brl: float = 0.0
    recommended_refund_brl: float = 0.0


class CaseOutput(BaseModel):
    case_id: str
    assessment: Assessment
    affected_entities: AffectedEntities
    root_cause_analysis: RootCauseAnalysis
    evidence_ids: list[str] = []
    financial_resolution: FinancialResolution
    resolution_actions: list[str] = []
