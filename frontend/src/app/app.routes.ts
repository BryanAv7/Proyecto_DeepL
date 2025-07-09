import { Routes } from '@angular/router';
import { ValidacionComponent } from './validacion/validacion.component';
import { DashboardComponent } from './dashboard/dashboard.component';

export const routes: Routes = [
  { path: '', component: ValidacionComponent },
  { path: 'dashboard', component: DashboardComponent },
  { path: '**', redirectTo: '' },
];
